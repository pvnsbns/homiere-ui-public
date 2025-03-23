import streamlit as st
import pandas as pd
import pydeck as pdk
import requests
from PIL import Image
import boto3
import io
import json
import boto3
import openai
from openai import OpenAI

# Page config
st.set_page_config(
    page_title="Homiere - Property Details",
    page_icon="./images/homiere_logo.png",
    layout="wide",
)

# Sidebar setup
st.sidebar.image("./images/homiere_logo.png", width=300)
st.sidebar.markdown("---")

# Display basic interactions
st.sidebar.markdown("""
### Basic Interactions
- **Chat with Homiere**: Ask Homiere anything about the current property!
""")
st.sidebar.markdown("---")

if st.sidebar.button("Return to Main Page"):
    st.session_state.clicked = False
    st.session_state.messages = []
    st.switch_page("main.py")

# Function to load map_data if not present
def load_map_data():
    if "map_data" not in st.session_state or st.session_state.map_data is None:
        try:
            s3_client = boto3.client(
                "s3",
                aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
                aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
                region_name=st.secrets["REGION_NAME"]
            )
            bucket_name = "homierebucket"
            processed_tabular_key = "processed_tabular.parquet"
            response = s3_client.get_object(Bucket=bucket_name, Key=processed_tabular_key)
            parquet_data = response["Body"].read()
            selected_columns = ['Geo Latitude', 'Geo Longitude', 'List Number', 'Address', 'City', 'List Price', 'Total Bedrooms', 'Total Bathrooms', 'Photo URL']
            st.session_state.map_data = pd.read_parquet(io.BytesIO(parquet_data), engine="pyarrow", columns=selected_columns)
        except Exception as e:
            st.error(f"Error loading map data: {e}")
            st.session_state.map_data = pd.DataFrame()

# Load map_data if necessary
load_map_data()

# Helper function for safe integer formatting
def safe_int(value):
    try:
        return int(float(value)) if pd.notna(value) else 0
    except:
        return 0

# Check for selected property
if "selected_listing_id" not in st.session_state:
    st.error("No property selected. Please go back and select a property from the search results.")
    if st.button("Go to Search Page"):
        st.switch_page("main.py")
else:
    listing_id = st.session_state.selected_listing_id
    if st.session_state.map_data is not None and not st.session_state.map_data.empty:
        property_row = st.session_state.map_data[st.session_state.map_data['List Number'] == listing_id]
        if not property_row.empty:
            property_data = property_row.iloc[0].to_dict()
            summary = st.session_state.get("selected_property_summary", "No summary available")
            summary = st.session_state.selected_property_summary 
            relevance = st.session_state.get("selected_property_relevance", "No relevance information available")
            
            # Header with back button
            col1, col2 = st.columns([1, 6])
            with col1:
                if st.button("← Back to Search"):
                    st.session_state.clicked = False
                    st.session_state.messages = []
                    st.switch_page("main.py")
            with col2:
                st.title("Property Details")
            
            # Main content
            main_col1, main_col2 = st.columns([3, 2])
            
            with main_col1:
                st.header(f"{property_data.get('Address', 'No Address')}")
                st.subheader(f"{property_data.get('City', 'No City')}")
                st.markdown(f"### ${safe_int(property_data.get('List Price', 0)):,}")
                st.markdown(f"**{safe_int(property_data.get('Total Bedrooms', 0))} beds | {safe_int(property_data.get('Total Bathrooms', 0))} baths**")
                st.markdown("### Property Summary")
                st.markdown(f"*{summary}*")
                st.markdown("### Relevance to Your Search")
                st.markdown(f"{relevance}")
            
            with main_col2:
                st.markdown("### Property Image")
                photo_url = property_data.get('Photo URL', '')
                if photo_url and photo_url.strip():
                    try:
                        st.image(photo_url, width=400)
                    except Exception:
                        st.image("https://i.imgur.com/Pj9k2Mn.png", width=400, caption="Image not available")
                else:
                    st.image("https://i.imgur.com/Pj9k2Mn.png", width=400, caption="Image not available")
                
                st.markdown("### Location")
                if 'Geo Latitude' in property_data and 'Geo Longitude' in property_data:
                    try:
                        lat = float(property_data['Geo Latitude'])
                        lon = float(property_data['Geo Longitude'])
                        map_data = pd.DataFrame([{
                            'lat': lat,
                            'lon': lon,
                            'tooltip': f"{property_data.get('Address', 'No Address')}, {property_data.get('City', 'No City')}"
                        }])
                        view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=15)
                        layer = pdk.Layer(
                            'ScatterplotLayer',
                            map_data,
                            get_position=['lon', 'lat'],
                            get_radius=100,
                            get_fill_color=[255, 0, 0, 200],
                            pickable=True
                        )
                        r = pdk.Deck(layers=[layer], initial_view_state=view_state, map_style='mapbox://styles/mapbox/light-v9', tooltip={"text": "{tooltip}"})
                        st.pydeck_chart(r, use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not display map: {e}")
                else:
                    st.warning("Location coordinates not available.")
        else:
            st.error(f"Property with List Number {listing_id} not found in data.")
     
    else:
        st.error("Property data not available. Please try again.")

def generate_prompt(selected_listing_id, query, generation_length_words=100):
    
    list_number_to_text = dict(zip(st.session_state.text_data['List Number'], st.session_state.text_data['enriched_text']))
    listing_description = list_number_to_text.get(selected_listing_id, "Listing description not found.")

    prompt = f"""
    User query: {query}
    Listing description: {listing_description}

    **Instructions:**  
    - Answer **directly and concisely** without prefacing or summarizing.  
    - Do **NOT** use phrases like "The final answer is" or "Based on the listing" or "Answer".  
    - Provide the response in **one clear sentence** without extra details.  
    - Avoid repetition, unnecessary context, or explanations.  
    - Keep the response **under {generation_length_words} words**.  
    - Escape all dollar signs as \$.  
    """

    return prompt

    
def initialize_bedrock_client():
    return boto3.client('bedrock-runtime', 
                        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID_BEDROCK"],
                        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY_BEDROCK"],
                        region_name=st.secrets["REGION_NAME"])

def call_llm(prompt, llm_model='llama-3-2', max_output_tokens=8000):
    if llm_model == 'llama-3-2':
        bedrock_client = initialize_bedrock_client()
        model_arn = "arn:aws:bedrock:us-west-2:626635417248:inference-profile/us.meta.llama3-2-3b-instruct-v1:0"
        try:
            request_payload = {
                "prompt": prompt,
                "max_gen_len": max_output_tokens,  # Adjust the max number of generated tokens as needed
            }
            # Properly format the body as a byte-string
            json_payload = json.dumps(request_payload).encode('utf-8')
            response = bedrock_client.invoke_model(
                modelId=model_arn,
                body=json_payload,  # Use the encoded JSON payload
                contentType="application/json"
            )
            if 'body' in response:
                # decode the response body
                response_body = json.loads(response['body'].read().decode('utf-8'))
                # access key for the generated text
                generated_text = response_body.get("generation", None)
                if generated_text:
                    return generated_text.strip()
                else:
                    print("ERROR: 'generation' key missing in response.")
                    return None
            else:
                print("ERROR: 'body' key is missing in response.")
                return None
        except Exception as e:
            print(f"ERROR: {e}")
            return None
    elif llm_model == 'llama-3-3':
        bedrock_client = initialize_bedrock_client()
        model_arn = "arn:aws:bedrock:us-west-2:626635417248:inference-profile/us.meta.llama3-3-70b-instruct-v1:0"
        try:
            request_payload = {
                "prompt": prompt,
                "max_gen_len": max_output_tokens,  # Adjust the max number of generated tokens as needed
            }
            # Properly format the body as a byte-string
            json_payload = json.dumps(request_payload).encode('utf-8')
            response = bedrock_client.invoke_model(
                modelId=model_arn,
                body=json_payload,  # Use the encoded JSON payload
                contentType="application/json"
            )
            if 'body' in response:
                # decode the response body
                response_body = json.loads(response['body'].read().decode('utf-8'))
                # access key for the generated text
                generated_text = response_body.get("generation", None)
                if generated_text:
                    return generated_text.strip()
                else:
                    print("ERROR: 'generation' key missing in response.")
                    return None
            else:
                print("ERROR: 'body' key is missing in response.")
                return None
        except Exception as e:
            print(f"ERROR: {e}")
            return None
    elif llm_model == 'gpt-4-turbo':
        try:
            response = openai.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": prompt}],  # Provide the prompt inside the messages field
                max_tokens=max_output_tokens,  # Adjust the max number of generated tokens as needed
                temperature=0.7,  # Optional: adjust creativity (lower for more focused responses)
            )
            generated_text = response.choices[0].message.content  # extract the generated text from the response
            return generated_text.strip()
        except Exception as e:
            print(f"ERROR: {e}")
            return None
    elif llm_model == 'gpt-3.5-turbo':
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],  # Provide the prompt inside the messages field
                max_tokens=max_output_tokens,  # Adjust the max number of generated tokens as needed
                temperature=0.7,  # Optional: adjust creativity (lower for more focused responses)
            )
            generated_text = response.choices[0].message.content  # extract the generated text from the response
            return generated_text.strip()
        except Exception as e:
            print(f"ERROR: {e}")
            return None
    elif llm_model == 'gpt-4o-mini':
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}], # Provide the prompt inside the messages field
                max_tokens=max_output_tokens, # Adjust the max number of generated tokens as needed
                temperature=0.7, # Optional: adjust creativity (lower for more focused responses)
            )
            generated_text = response.choices[0].message.content # extract the generated text from the response
            return generated_text.strip()
        except Exception as e:
            print(f"ERROR: {e}")
            return None
        

if "messages" not in st.session_state:
    st.session_state.messages = []

if query := st.chat_input("Ask anything about the current property..."):
    st.session_state.messages.append({"role": "user", "content": query})

    prompt = generate_prompt(st.session_state.selected_listing_id, query, generation_length_words=100)
    llm_response = call_llm(prompt, llm_model='llama-3-3', max_output_tokens=1000)

    st.session_state.messages.append({"role": "assistant", "content": llm_response})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])