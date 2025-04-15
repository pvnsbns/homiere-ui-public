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
import urllib.parse
import ast

if  "property_data" not in st.session_state:
    st.session_state.property_data = None
if "messages" not in st.session_state:
    st.session_state.messages = []

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
### Property Insights
- **Engage with Homiere**: Get detailed information about the property on a wide range of topics, including:
  1. Nearby Schools and their ratings
  2. Walk, Bike, and Transit Accessibility Scores
  3. Air Quality and Pollution Data
  4. Public Safety
  5. And many other key property features
""")
st.sidebar.markdown("---")

# Add custom CSS to match the logo color
st.markdown("""
<style>
    /* Main colors from the HOMIERE logo */
    :root {
        --primary-dark: #722F37;    /* Deep burgundy */
        --primary-medium: #8B4539;  /* Medium burgundy/brown */
        --primary-light: #A06A45;   /* Light brown */
        --accent-cream: #F9F6F2;    /* Light cream background */
        --text-dark: #2C2C2C;       /* Dark text */
    }
    
    /* Global text colors */
    body {
        color: var(--text-dark);
    }
    
    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: var(--primary-dark);
    }
    
    /* Buttons */
    .stButton > button {
        background-color: var(--primary-dark);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: var(--primary-light);
    }
    .stButton > button:active {
        background-color: var(--primary-medium);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--accent-cream);
        border-right: 1px solid #E0D1C0;
    }
    [data-testid="stSidebar"] hr {
        border-color: var(--primary-light);
    }
    
    /* Cards and info boxes */
    .info-box {
        background-color: var(--accent-cream);
        border-left: 4px solid var(--primary-dark);
        padding: 15px;
        border-radius: 4px;
        margin: 15px 0;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: var(--accent-cream);
        border-radius: 4px 4px 0 0;
        padding: 10px 16px;
        color: var(--primary-medium);
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: var(--primary-dark);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: var(--primary-dark);
        font-weight: bold;
    }
    
    /* Inputs */
    input[type="text"], input[type="number"], select, textarea {
        border-radius: 4px;
        border: 1px solid #E0D1C0;
    }
    input[type="text"]:focus, input[type="number"]:focus, select:focus, textarea:focus {
        border-color: var(--primary-medium);
        box-shadow: 0 0 0 1px var(--primary-light);
    }
    
    /* Sliders */
    .stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] {
        background-color: var(--primary-dark);
        color: white;
    }
    
    /* Dividers with gradient */
    hr {
        height: 2px;
        border: none;
        background: linear-gradient(to right, var(--primary-dark), var(--primary-light), var(--accent-cream));
    }
            
    /* Make the chat input more prominent */
    .stChatInput {
        border: 2px solid #722F37 !important;  /* Primary dark burgundy color from your theme */
        border-radius: 10px !important;
        padding: 8px !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1) !important;
        transition: all 0.3s ease !important;
    }

    /* Add a slight glow effect on hover */
    .stChatInput:hover {
        box-shadow: 0 6px 12px rgba(114, 47, 55, 0.2) !important;
        border-color: #A06A45 !important;  /* Primary light color from your theme */
    }
</style>
""", unsafe_allow_html=True)

if st.sidebar.button("Return to Main Page"):
    st.session_state.clicked = False
    st.session_state.messages = []
    st.switch_page("main.py")

# Helper function for safe integer formatting
def safe_int(value):
    try:
        return int(float(value)) if pd.notna(value) else 0
    except:
        return 0

def generate_prompt(selected_listing_id, query, generation_length_words=100):
    
    list_number_to_text = dict(zip(st.session_state.text_data['List Number'], st.session_state.text_data['enriched_text']))
    listing_description = list_number_to_text.get(selected_listing_id, "Listing description not found.")

    prompt = f"""
    User query: {query}
    Listing description: {listing_description}

    **Instructions:**  
    - Answer **directly and concisely** without prefacing or summarizing. You may be conversational as a real estate agent with **one clear sentence** response.
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
        
def load_property_data():
    """Load additional property data (processed_tabular.parquet) from S3."""
    if st.session_state.property_data is None:
        try:
            s3_client = boto3.client(
                "s3",
                aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
                aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
                region_name=st.secrets["REGION_NAME"]
            )
            bucket_name = "homierebucket"
            processed_tabular_key = "processed_tabular.parquet"
            processed_tabular_response = s3_client.get_object(Bucket=bucket_name, Key=processed_tabular_key)
            processed_tabular_parquet = processed_tabular_response["Body"].read()
            
            selected_columns = ['List Number', 'Year Built', 'Public Remarks', 'Parsed Features', 'property_condition', 'price_per_sqft', 'price_diff_from_city_avg', 'monthly_mortgage', 'days_on_market', 'type',
                                'Violent Crime per 1000','Violent Crime Category', 'Property Crime per 1000', 'Property Crime Category',
                                'walk_score', 'bike_score', 'transit_score', 'walk_category', 'bike_category', 'transit_category',
                                'Elementary_School_Rating', 'Middle_School_Rating', 'High_School_Rating']
            st.session_state.property_data = pd.read_parquet(io.BytesIO(processed_tabular_parquet), engine="pyarrow", columns=selected_columns)
        except Exception as e:
            st.error(f"Error loading additional property data from S3: {e}")
            st.session_state.property_data = pd.DataFrame()

if "property_data" not in st.session_state:
    load_property_data()
      
# Check if property is selected
if "selected_listing_id" not in st.session_state:
    st.error("No property selected. Please go back and select a property from the search results.")
    if st.button("Go to Search Page"):
        st.switch_page("main.py")
else:
    listing_id = st.session_state.selected_listing_id
    if st.session_state.map_data is not None and not st.session_state.map_data.empty:
        property_row = st.session_state.map_data[st.session_state.map_data['List Number'] == listing_id]
        additional_property_row = st.session_state.property_data[st.session_state.property_data['List Number'] == listing_id]
        if not property_row.empty:          
            if st.button("← Back to Search"):
                st.session_state.clicked = False
                st.session_state.messages = []
                st.switch_page("main.py")
            
            property_data = property_row.iloc[0].to_dict()
            additional_property_data = additional_property_row.iloc[0].to_dict()
            summary = st.session_state.get("selected_property_summary", "No summary available")
            relevance = st.session_state.get("selected_property_relevance", "No relevance information available")

            address = property_data.get('Address', 'No Address')
            city = property_data.get('City', 'No City')
            state = property_data.get('State', 'No State')
            zip_code = property_data.get('Zip', 'No Zip')
            list_price = safe_int(property_data.get('List Price', 0))
            bedrooms = safe_int(property_data.get('Total Bedrooms', 0))
            bathrooms = safe_int(property_data.get('Total Bathrooms', 0))
            sqft = safe_int(property_data.get('Total Sqft', 0))

            # For additional data loading for property details
            year_built = safe_int(additional_property_data.get('Year Built', 0))
            public_remarks = additional_property_data.get('Public Remarks', 'No Remarks on Property')
            property_condition = additional_property_data.get('property_condition', 'No Condition').title()
            price_per_sqft = round(safe_int(additional_property_data.get('price_per_sqft', 0)), 2)
            price_diff_from_city_avg = round(safe_int(additional_property_data.get('price_diff_from_city_avg', 0)), 2)
            monthly_mortgage = round(safe_int(additional_property_data.get('monthly_mortgage', 0)), 2)
            days_on_market = safe_int(additional_property_data.get('days_on_market', 0))
            property_type = additional_property_data.get('type', 'No Type')
            features = additional_property_data.get('Parsed Features', {})

            full_address = f"{address}, {city}, {state} {zip_code}"
            google_maps_url = f"https://www.google.com/maps/place/{urllib.parse.quote(full_address)}"
            
            col1, col2 = st.columns([5, 2], gap="small")

            with col1:
                st.title("Property Overview")
                overview_tab, neighborhood_tab, property_tab, features_tab, cost_tab = \
                    st.tabs(["OVERVIEW", "NEIGHBORHOOD", "INTERIOR/EXTERIOR", "ADDITIONAL FEATURES", "COST"])
                css = '''
                <style>
                    .stTabs [data-baseweb="tab-list"] button {
                        margin-right: 10px;  /* Adds space between the tabs */
                    }
                    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
                        font-size: 1.05rem;
                    }
                </style>
                '''

                st.markdown(css, unsafe_allow_html=True)

                with overview_tab: 
                    inside_col1, inside_col2 = st.columns([3, 2])
                    
                    with inside_col1:
                        st.markdown(f"## <span style='font-weight: bold;'>{address}</span>", unsafe_allow_html=True)
                        st.markdown(f"### {city}, {state} {zip_code}")
                        st.markdown(f"#### <span style='font-size: 20px;'><strong>Home Type:</strong> {property_type.title()}</span>", unsafe_allow_html=True)
                        st.markdown(f"[View on Google Maps]({google_maps_url})", unsafe_allow_html=True)

                    with inside_col2:
                        st.markdown(f"## <span style='color: green; font-size: 48px;'>${list_price:,}</span>", unsafe_allow_html=True)
                        st.markdown(f"#### <span style='font-size: 20px;'>**{bedrooms} Beds** | **{bathrooms} Baths** | **{sqft} Sqft**</span>", unsafe_allow_html=True)
                        st.markdown(f"#### <span style='font-size: 20px;'>**{property_condition}** | **Built in {year_built}**</span>", unsafe_allow_html=True)

                    st.markdown("<hr>", unsafe_allow_html=True)

                    st.markdown("### **Property Description**")
                    st.write(f"{public_remarks}")
                    
                    st.markdown("<hr>", unsafe_allow_html=True)
                
                with neighborhood_tab:
                    # Helper functions
                    def get_category_color(category, is_crime=False):
                        """Get appropriate color based on category text.
                        For crime categories, is_crime=True will reverse the logic (lower is better)."""
                        if not category or pd.isna(category):
                            return "#9E9E9E"  # Gray for missing data
                            
                        category = str(category).lower().strip()
    
                        # Define mapping for categories
                        exact_category_map = {
                            # Score 5 - Excellent (Green)
                            "walker's paradise": 5,
                            "rider's paradise": 5,
                            "biker's paradise": 5,
                            "very low violent crime risk": 5,
                            "very low property crime risk": 5,
                            
                            # Score 4 - Very Good (Light Green)
                            "very walkable": 4,
                            "very bikeable": 4,
                            "excellent transit": 4,
                            "above average": 4,
                            "lower violent crime risk": 4,
                            "lower property crime risk": 4,
                            
                            # Score 3 - Good (Yellow)
                            "somewhat walkable": 3,
                            "bikeable": 3,
                            "good transit": 3,
                            "average": 3,
                            "average violent crime risk": 3,
                            "average property crime risk": 3,
                            
                            # Score 2 - Fair (Orange)
                            "somewhat bikeable": 2,
                            "some transit": 2,
                            "mostly car-dependent": 2,
                            "below average": 2,
                            "higher violent crime risk": 2,
                            "higher property crime risk": 2,
                            
                            # Score 1 - Poor (Red)
                            "completely car-dependent": 1,
                            "minimal transit": 1,
                            "poor": 1,
                            "very high violent crime risk": 1,
                            "very high property crime risk": 1
                        }
                        
                        # Colors from best to worst
                        colors = {
                            5: "#4CAF50",  # Green
                            4: "#8BC34A",  # Light Green
                            3: "#FFC107",  # Yellow
                            2: "#FF9800",  # Orange
                            1: "#F44336",  # Red
                            0: "#9E9E9E"   # Gray (default)
                        }
                        
                        if category in exact_category_map:
                            return colors[exact_category_map[category]]
                        
                        # If no exact match, try to match the most specific phrases first
                        # We'll check from most specific (longest) to least specific
                        sorted_categories = sorted(exact_category_map.keys(), key=len, reverse=True)
                        
                        for key in sorted_categories:
                            if key in category:
                                return colors[exact_category_map[key]]
                        
                        # If no match found, return default gray
                        return colors[0]
                    
                    def safe_int(value):
                        try:
                            if pd.isna(value):
                                return "N/A"
                            return int(value)
                        except:
                            return "N/A"
                    
                    def safe_float(value, decimals=1):
                        try:
                            if pd.isna(value):
                                return "N/A"
                            return round(float(value), decimals)
                        except:
                            return "N/A"
                    
                    # Create sections for walkability, safety, and schools
                    st.markdown("### **Walkability & Transportation**")
                    
                    # Create a row of 3 columns for walkability scores
                    walk_col1, walk_col2, walk_col3 = st.columns(3)
                    
                    # Extract walkability data
                    walk_score = safe_int(additional_property_data.get('walk_score', "N/A"))
                    walk_category = additional_property_data.get('walk_category', "No data available")
                    if pd.isna(walk_category):
                        walk_category = "No data available"
                        
                    bike_score = safe_int(additional_property_data.get('bike_score', "N/A"))
                    bike_category = additional_property_data.get('bike_category', "No data available")
                    if pd.isna(bike_category):
                        bike_category = "No data available"
                        
                    transit_score = safe_int(additional_property_data.get('transit_score', "N/A"))
                    transit_category = additional_property_data.get('transit_category', "No data available")
                    if pd.isna(transit_category):
                        transit_category = "No data available"
                    
                    # Display walk score and category
                    with walk_col1:
                        walk_color = get_category_color(walk_category)
                        st.markdown(f"""
                        <div style="background-color: #f7f7f7; padding: 15px; border-radius: 5px; text-align: center;">
                            <h4 style="margin-top: 0;">Walk Score</h4>
                            <div style="font-size: 36px; font-weight: bold; color: {walk_color};">{walk_score}</div>
                            <div style="margin-top: 5px; color: {walk_color}; font-weight: bold;">
                                {walk_category}
                            </div>
                            <div style="height: 8px; background-color: {walk_color}; width: 100%; border-radius: 4px; margin-top: 10px;"></div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Display transit score and category
                    with walk_col2:
                        transit_color = get_category_color(transit_category)
                        st.markdown(f"""
                        <div style="background-color: #f7f7f7; padding: 15px; border-radius: 5px; text-align: center;">
                            <h4 style="margin-top: 0;">Transit Score</h4>
                            <div style="font-size: 36px; font-weight: bold; color: {transit_color};">{transit_score}</div>
                            <div style="margin-top: 5px; color: {transit_color}; font-weight: bold;">
                                {transit_category}
                            </div>
                            <div style="height: 8px; background-color: {transit_color}; width: 100%; border-radius: 4px; margin-top: 10px;"></div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Display bike score and category
                    with walk_col3:
                        bike_color = get_category_color(bike_category)
                        st.markdown(f"""
                        <div style="background-color: #f7f7f7; padding: 15px; border-radius: 5px; text-align: center;">
                            <h4 style="margin-top: 0;">Bike Score</h4>
                            <div style="font-size: 36px; font-weight: bold; color: {bike_color};">{bike_score}</div>
                            <div style="margin-top: 5px; color: {bike_color}; font-weight: bold;">
                                {bike_category}
                            </div>
                            <div style="height: 8px; background-color: {bike_color}; width: 100%; border-radius: 4px; margin-top: 10px;"></div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("<hr>", unsafe_allow_html=True)
                    
                    # Safety Section
                    st.markdown("### **Neighborhood Safety**")
                    
                    safety_col1, safety_col2 = st.columns(2)
                    
                    # Get crime data (both numerical and categorical)
                    violent_crime_rate = safe_float(additional_property_data.get('Violent Crime per 1000', "N/A"))
                    violent_crime_category = additional_property_data.get('Violent Crime Category', "No data available")
                    if pd.isna(violent_crime_category):
                        violent_crime_category = "No data available"
                    
                    property_crime_rate = safe_float(additional_property_data.get('Property Crime per 1000', "N/A"))
                    property_crime_category = additional_property_data.get('Property Crime Category', "No data available")
                    if pd.isna(property_crime_category):
                        property_crime_category = "No data available"
                    
                    # Display violent crime information
                    with safety_col1:
                        violent_crime_color = get_category_color(violent_crime_category, is_crime=True)
                        st.markdown(f"""
                        <div style="background-color: #f7f7f7; padding: 15px; border-radius: 5px; height: 100%;">
                            <h4 style="margin-top: 0;">Violent Crime</h4>
                            <div style="display: flex; align-items: center; justify-content: space-between;">
                                <div>
                                    <p style="margin-bottom: 5px;">Risk Level:</p>
                                    <p style="color: {violent_crime_color}; font-weight: bold; font-size: 18px; margin-top: 0;">{violent_crime_category}</p>
                                    <p style="margin-bottom: 0;">Rate: {violent_crime_rate} per 1,000 residents</p>
                                </div>
                                <div style="width: 70px; height: 70px; border-radius: 50%; background-color: {violent_crime_color}; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
                                    {"Low" if "low" in str(violent_crime_category).lower() else 
                                    "High" if "high" in str(violent_crime_category).lower() else
                                    "Avg" if "average" in str(violent_crime_category).lower() else "N/A"}
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Display property crime information
                    with safety_col2:
                        property_crime_color = get_category_color(property_crime_category, is_crime=True)
                        st.markdown(f"""
                        <div style="background-color: #f7f7f7; padding: 15px; border-radius: 5px; height: 100%;">
                            <h4 style="margin-top: 0;">Property Crime</h4>
                            <div style="display: flex; align-items: center; justify-content: space-between;">
                                <div>
                                    <p style="margin-bottom: 5px;">Risk Level:</p>
                                    <p style="color: {property_crime_color}; font-weight: bold; font-size: 18px; margin-top: 0;">{property_crime_category}</p>
                                    <p style="margin-bottom: 0;">Rate: {property_crime_rate} per 1,000 residents</p>
                                </div>
                                <div style="width: 70px; height: 70px; border-radius: 50%; background-color: {property_crime_color}; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
                                    {"Low" if "low" in str(property_crime_category).lower() else 
                                    "High" if "high" in str(property_crime_category).lower() else
                                    "Avg" if "average" in str(property_crime_category).lower() else "N/A"}
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("<hr>", unsafe_allow_html=True)
                    
                    # Schools Section
                    st.markdown("### **Nearby Schools**")

                    # Get school ratings with default for missing values
                    elem_rating = additional_property_data.get('Elementary_School_Rating', "No data available")
                    if pd.isna(elem_rating):
                        elem_rating = "No data available"

                    middle_rating = additional_property_data.get('Middle_School_Rating', "No data available")
                    if pd.isna(middle_rating):
                        middle_rating = "No data available"

                    high_rating = additional_property_data.get('High_School_Rating', "No data available")
                    if pd.isna(high_rating):
                        high_rating = "No data available"

                    # Display school information
                    school_col1, school_col2, school_col3 = st.columns(3)

                    # Function to get color for school ratings
                    def get_school_rating_color(rating):
                        if not rating or pd.isna(rating) or rating == "No data available":
                            return "#9E9E9E"  # Gray for missing data
                            
                        rating = str(rating).lower()
                        if "above average" in rating:
                            return "#4CAF50"  # Green
                        elif "average" in rating:
                            return "#FFC107"  # Yellow
                        elif "below average" in rating:
                            return "#FF9800"  # Orange
                        else:
                            return "#9E9E9E"  # Gray for other cases

                    # Create a standardized school card display
                    def school_card(column, title, rating):
                        color = get_school_rating_color(rating)
                        column.markdown(f"""
                        <div style="background-color: #f7f7f7; padding: 15px; border-radius: 5px; text-align: center; height: 120px;">
                            <h4 style="margin-top: 0;">{title}</h4>
                            <div style="height: 50px; display: flex; align-items: center; justify-content: center;">
                                <span style="font-size: 18px; font-weight: bold; color: {color};">{rating}</span>
                            </div>
                            <div style="height: 8px; background-color: {color}; width: 100%; border-radius: 4px; margin-top: 5px;"></div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Display all three schools with the same card function
                    school_card(school_col1, "Elementary School", elem_rating)
                    school_card(school_col2, "Middle School", middle_rating)
                    school_card(school_col3, "High School", high_rating)

                with property_tab:
                    property_features = {
                        "Interior Features": [
                            "Architectural Style", "Levels", "Kitchen Features", "Bathroom Features", "Flooring",
                            "Interior Features", "Fireplace Features", "Rooms", "Fireplace Location", "Eating Area", 
                            "Common Walls", "Miscellaneous Information", "Structure & Foundation", 
                            "Windows", "Structure Type", "Basement"
                        ],
                        "Exterior Features": [
                            "Lot Features", "Community Features", "Lot Location", "Lot Description", "View", 
                            "Waterfront Features", "View Type", "Exterior Features", "Patio Features", "Exterior Construction",
                            "Water Source", "Road Frontage Type", "Road Surface Type", "Irrigation", "Sprinklers",
                            "Roof", "Window Features", "Door Features", "Patio and Porch Features" 
                        ]
                    }

                    category_icons_property = {
                        "Interior Features": '🏠',
                        "Exterior Features": '🌳'
                    }

                    try:
                        if isinstance(features, str):
                            features = ast.literal_eval(features)

                        for category, feature_list in property_features.items():
                            relevant_features = [feature for feature in feature_list if feature in features]

                            if relevant_features:
                                st.markdown(f"""
                                    <h3 style='text-align: left; color: #000000; margin-bottom: 10px;'>
                                        {category_icons_property.get(category, '🔹')} <strong>{category}</strong>
                                    </h3>
                                """, unsafe_allow_html=True)
                                num_columns = 3
                                columns = st.columns(num_columns)
                                column_index = 0

                                for feature in relevant_features:
                                    values = features[feature]
                                    formatted_feature = ', '.join(values)
                                    columns[column_index].markdown(
                                        f"<div style='background-color: #f7f7f7; padding: 10px; border-radius: 5px; margin-bottom: 20px;'>"
                                        f"<strong>{feature}:</strong> {formatted_feature}</div>", 
                                        unsafe_allow_html=True
                                    )
                                    column_index = (column_index + 1) % num_columns
                                st.markdown("<hr>", unsafe_allow_html=True)

                    except Exception as e:
                        st.write(f"Error processing features: {e}")

                with features_tab:
                    utility_amenity_features = {
                        "Energy": [
                            "Heating", "Cooling", "Heating Type", "Heating Fuel", "Cooling Type", "Fireplace Fuel",
                            "Green Energy Generation", "Green Energy Efficient",
                            "Water Heater Feature", "Utilities", "Electric", "Sewer", "Water"
                        ],
                        "Accessibility": [
                            "Cooking Appliances", 
                            "HOA Information", "HOA 1 Frequency", "HOA 2 Frequency", "HOA 3 Frequency", "Association Amenities",
                            "Association Fees Include", "Disclosures", "Park Information", "Other Structures", "Other Property Features",
                            "Parking", "Parking Features", "Parking Type", "Parking Spaces/Information", 
                            "Accessibility Features", "Disability Access"
                        ],
                        "Mobile & RV Features": [
                            "Mobile Home Type", "Mobile/Manufactured Info", "RV Park Information", "RV Hook-Ups", 
                            "RV Spot Fuel Type", "Volt 220 Location"
                        ]
                    }

                    category_icons_combined = {
                        "Energy": '⚡',
                        "Accessibility": '♿',
                        "Mobile/RV Features": '🚐'
                    }

                    try:
                        if isinstance(features, str):
                            features = ast.literal_eval(features)

                        for category, feature_list in utility_amenity_features.items():
                            relevant_features = [feature for feature in feature_list if feature in features]

                            if relevant_features:
                                st.markdown(f"""
                                    <h3 style='text-align: left; color: #000000; margin-bottom: 10px;'>
                                        {category_icons_combined.get(category, '🔹')} <strong>{category}</strong>
                                    </h3>
                                """, unsafe_allow_html=True)       
                                num_columns = 3
                                columns = st.columns(num_columns)
                                column_index = 0

                                for feature in relevant_features:
                                    values = features[feature]
                                    formatted_feature = ', '.join(values)
                                    columns[column_index].markdown(
                                        f"<div style='background-color: #f7f7f7; padding: 10px; border-radius: 5px; margin-bottom: 20px;'>"
                                        f"<strong>{feature}:</strong> {formatted_feature}</div>", 
                                        unsafe_allow_html=True
                                    )
                                    column_index = (column_index + 1) % num_columns
                                st.markdown("<hr>", unsafe_allow_html=True)

                    except Exception as e:
                        st.write(f"Error processing features: {e}")

                with cost_tab:
                    # Function to calculate mortgage
                    def calculate_monthly_mortgage(principal, annual_interest_rate, loan_term_years):
                        monthly_interest_rate = annual_interest_rate / 12 / 100
                        number_of_payments = loan_term_years * 12
                        monthly_payment = (principal * monthly_interest_rate) / (1 - (1 + monthly_interest_rate) ** -number_of_payments)
                        return monthly_payment

                    st.markdown(f"## <span style='font-size: 30px;'>**Listing Price:** <span style='color: green;'>{list_price:,}</span> | **Price per Sqft:** <span style='color: green;'>{price_per_sqft:,.2f}</span></span>", unsafe_allow_html=True)

                    # Display price difference from the city average
                    if price_diff_from_city_avg > 0:
                        st.markdown(f"#### <span style='font-size: 20px;'>The price per sqft for this property is <span style='color: red;'><strong>${price_diff_from_city_avg:,.2f}</strong></span> higher than the average city price per sqft price.</span>", unsafe_allow_html=True)
                    elif price_diff_from_city_avg < 0:
                        st.markdown(f"#### <span style='font-size: 20px;'>The price per sqft for this property is <span style='color: green;'><strong>${abs(price_diff_from_city_avg):,.2f}</strong></span> lower than the average city price per sqft price.</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"#### <span style='font-size: 20px;'>The price per sqft for this property is exactly the same as the average city price per sqft price.</span>", unsafe_allow_html=True)


                    st.markdown("<hr>", unsafe_allow_html=True)

                    st.markdown("### **Mortgage Calculator**")

                    cost_col1, cost_col2, cost_col3 = st.columns([2, 1, 1])

                    with cost_col1:
                        downpayment_percentage = st.slider("Down Payment Percentage", 0, 100, 20)

                    with cost_col2:
                        annual_interest_rate = st.number_input("Annual Interest Rate (%)", min_value=0.0, value=6.0, step=0.1)

                    with cost_col3:
                        loan_term_years = st.number_input("Loan Term (Years)", min_value=1, value=30, step=1)

                    # Calculate Down Payment, Loan Amount, and Monthly Mortgage
                    downpayment_amount = (downpayment_percentage / 100) * list_price
                    loan_amount = list_price - downpayment_amount
                    monthly_mortgage = calculate_monthly_mortgage(loan_amount, annual_interest_rate, loan_term_years)

                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    calculation_col1, calculation_col2, calculation_col3 = st.columns(3)

                    with calculation_col1:
                        st.markdown(f"<div style='background-color: #f7f7f7; padding: 5px; border-radius: 5px; text-align: center;'><h4 style='color: #0073e6;'>Down Payment</h4><p style='font-size: 20px;'>${downpayment_amount:,.2f}</p></div>", unsafe_allow_html=True)

                    with calculation_col2:
                        st.markdown(f"<div style='background-color: #f7f7f7; padding: 5px; border-radius: 5px; text-align: center;'><h4 style='color: #0073e6;'>Loan Amount</h4><p style='font-size: 20px;'>${loan_amount:,.2f}</p></div>", unsafe_allow_html=True)

                    with calculation_col3:
                        st.markdown(f"<div style='background-color: #f7f7f7; padding: 5px; border-radius: 5px; text-align: center;'><h4 style='color: #0073e6;'>Monthly Mortgage</h4><p style='font-size: 20px;'>${monthly_mortgage:,.2f}</p></div>", unsafe_allow_html=True)


            # Right column for Property Image and Map
            with col2:
                # Display Property Image
                photo_url = property_data.get('Photo URL', '')
                
                # Check if a photo URL exists
                if photo_url and photo_url.strip():
                    try:
                        st.image(photo_url, width=400)
                    except Exception:
                        st.image("https://i.imgur.com/Pj9k2Mn.png", width=400, caption="Image not available")
                else:
                    st.image("https://i.imgur.com/Pj9k2Mn.png", width=400, caption="Image not available")
                
                # Display Map (Location)
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
                            get_radius=30,
                            get_fill_color=[255, 0, 0, 200],
                            pickable=True
                        )
                        r = pdk.Deck(layers=[layer], initial_view_state=view_state, map_style='mapbox://styles/mapbox/outdoors-v11', tooltip={"text": "{tooltip}"})
                        st.pydeck_chart(r, use_container_width=False, height=300, width=400)
                    except Exception as e:
                        st.error(f"Could not display map: {e}")
                else:
                    st.warning("Location coordinates not available.")

            if query := st.chat_input("Ask anything about the current property..."):
                st.session_state.messages.append({"role": "user", "content": query})

                prompt = generate_prompt(st.session_state.selected_listing_id, query, generation_length_words=100)
                llm_response = call_llm(prompt, llm_model='llama-3-3', max_output_tokens=1000)

                st.session_state.messages.append({"role": "assistant", "content": llm_response})

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

    else:
        st.error("Property data not available. Please try again.")
        