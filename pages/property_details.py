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
            
            selected_columns = ['List Number', 'Year Built', 'Parsed Features', 'property_condition', 'price_per_sqft', 'price_diff_from_city_avg', 'monthly_mortgage', 'days_on_market', 'type']
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
            property_condition = additional_property_data.get('property_condition', 'No Condition')
            price_per_sqft = round(safe_int(additional_property_data.get('price_per_sqft', 0)), 2)
            price_diff_from_city_avg = round(safe_int(additional_property_data.get('price_diff_from_city_avg', 0)), 2)
            monthly_mortgage = round(safe_int(additional_property_data.get('monthly_mortgage', 0)), 2)
            days_on_market = safe_int(additional_property_data.get('days_on_market', 0))
            property_type = additional_property_data.get('type', 'No Type')
            features = additional_property_data.get('Parsed Features', {})

            full_address = f"{address}, {city}, {state} {zip_code}"
            google_maps_url = f"https://www.google.com/maps/place/{urllib.parse.quote(full_address)}"
            
            col1, col2 = st.columns([3, 2], gap="large")

            with col1:
                st.title("Property Overview")
                overview_tab, property_tab, features_tab, cost_tab = \
                    st.tabs(["PROPERTY OVERVIEW", "PROPERTY INTERIOR/EXTERIOR", "ADDITIONAL FEATURES", "COST"])
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

                    st.markdown("### **Property Summary**")
                    st.write(f"{summary}")

                    st.markdown("<hr>", unsafe_allow_html=True)
                
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
        