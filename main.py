from utils.retrieval_pipeline import execute_retrieval

import json
import streamlit as st
import requests
import pandas as pd
import pydeck as pdk
import boto3
import io
import random
from io import StringIO
from PIL import Image


# Page config
st.set_page_config(
    page_title="Homiere - Your Personalized Home Search Assistant",
    page_icon="./images/homiere_logo.png",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        "Get help": "https://github.com/pvnsbns/homiere-ui/",
        "Report a bug": "https://github.com/pvnsbns/homiere-ui/",
        "About": """
            ## Homiere - Home Search Assistant
            ### Powered using GPT-3.5-turbo

            **GitHub**: https://github.com/pvnsbns/homiere-ui/

            Homiere is your AI assistant to search your perfect home!
        """
    }
)

# Sidebar setup
st.sidebar.image("./images/homiere_logo.png", width=300)
st.sidebar.markdown("---")

# Display basic interactions
st.sidebar.markdown("""
### Basic Interactions
- **Chat with Homiere**: Tell Homiere what you're looking for in a home, and let AI refine your search.
- **Optional Filter**: Use quick-start filters to narrow down your preferences and customize your results.
- **Interactive Map**: Explore available properties across Southern California by navigating the map.
""")
st.sidebar.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "map_data" not in st.session_state:
    st.session_state.map_data = None
if "text_data" not in st.session_state:
    st.session_state.text_data = None
if  "property_data" not in st.session_state:
    st.session_state.property_data = None
if "refresh_chat" not in st.session_state:
    st.session_state.refresh_chat = False
if "properties_df" not in st.session_state:
    st.session_state.properties_df = None
if "show_expander" not in st.session_state:
    st.session_state.show_expander = False
if "search_triggered" not in st.session_state:
    st.session_state.search_triggered = False
if "expander_shown" not in st.session_state:
    st.session_state.expander_shown = True
if "expander_state" not in st.session_state:
    st.session_state.expander_state = True  # Start expanded
if "selected_property" not in st.session_state:
    st.session_state.selected_property = None
if "selected_property_summary" not in st.session_state:
    st.session_state.selected_property_summary = None
if "selected_property_relevance" not in st.session_state:
    st.session_state.selected_property_relevance = None
if "selected_listing_id" not in st.session_state:
    st.session_state.selected_listing_id = None
if 'clicked' not in st.session_state:
    st.session_state.clicked = False
if 'conclusion' not in st.session_state:
    st.session_state.conclusion = ""
if 'clicked_property' not in st.session_state:
    st.session_state.clicked_property = ""
if 'listing_buttons_list' not in st.session_state:
    st.session_state.listing_buttons_list = []

# Function to load data from S3
# @st.cache_data(show_spinner=False)
def load_map_data():
    """Load map data (processed_tabular.parquet) from S3."""
    if st.session_state.map_data is None:
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
            
            selected_columns = ['Geo Latitude', 'Geo Longitude', 'List Number', 'Address', 'City', 'State', 'Zip', 'Total Sqft', 'List Price', 'Total Bedrooms', 'Total Bathrooms', 'Photo URL', 'caption']
            st.session_state.map_data = pd.read_parquet(io.BytesIO(processed_tabular_parquet), engine="pyarrow", columns=selected_columns)
        except Exception as e:
            st.error(f"Error loading map data from S3: {e}")
            st.session_state.map_data = pd.DataFrame()

def load_text_data():
    """Load text data (processed_text.parquet) from S3."""
    if st.session_state.text_data is None:
        try:
            s3_client = boto3.client(
                "s3",
                aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
                aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
                region_name=st.secrets["REGION_NAME"]
            )
            bucket_name = "homierebucket"
            processed_text_key = "processed_text.parquet"
            processed_text_response = s3_client.get_object(Bucket=bucket_name, Key=processed_text_key)
            processed_text_parquet = processed_text_response["Body"].read()
            st.session_state.text_data = pd.read_parquet(io.BytesIO(processed_text_parquet), engine="pyarrow")
        except Exception as e:
            st.error(f"Error loading text data from S3: {e}")
            st.session_state.text_data = pd.DataFrame()

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
            
            selected_columns = ['List Number', 'Year Built', 'Public Remarks', 'Parsed Features', 'property_condition', 'price_per_sqft', 'price_diff_from_city_avg', 'monthly_mortgage', 'days_on_market', 'type']
            st.session_state.property_data = pd.read_parquet(io.BytesIO(processed_tabular_parquet), engine="pyarrow", columns=selected_columns)
        except Exception as e:
            st.error(f"Error loading additional property data from S3: {e}")
            st.session_state.property_data = pd.DataFrame()

# Need to load map first, then can grab text data after so user doesn't have to wait for map to render
load_map_data()

def render_map(map_data):
    if not map_data.empty:
        map_data = pd.DataFrame([
            {
                'lat': row.get('Geo Latitude', 0),
                'lon': row.get('Geo Longitude', 0),
                'List Number': row.get('List Number', ''),
                'Address': row.get('Address', ''),
                'City': row.get('City', ''),
                'tooltip_text': f"{row.get('Address', '')}, {row.get('City', '')}\nList #: {row.get('List Number', '')}"
            }
            for _, row in map_data.iterrows()
        ])

        if len(map_data) > 0:
            center_lat = map_data['lat'].mean()
            center_lon = map_data['lon'].mean()
            
            # Calculate the bounds of the data
            lat_range = map_data['lat'].max() - map_data['lat'].min()
            lon_range = map_data['lon'].max() - map_data['lon'].min()
            
            # Calculate a dynamic zoom level based on the ranges
            max_range = max(lat_range, lon_range)
            if max_range > 0.5:  # Very spread out
                zoom_level = 8
            elif max_range > 0.2:
                zoom_level = 9
            elif max_range > 0.05:
                zoom_level = 10
            elif max_range > 0.01:
                zoom_level = 11
            else:  # Very close together
                zoom_level = 12
                
            # If there's only one point, use a closer zoom
            if len(map_data) == 1:
                zoom_level = 13
                
            # Scale point radius based on zoom level and number of points
            # Increased base radius for better visibility
            base_radius = 400  # Increased from 300
            
            # Make points significantly larger when there are few of them
            if len(map_data) <= 3:
                point_radius = base_radius * 2.0  # Much larger for very few points
            elif len(map_data) <= 10:
                point_radius = base_radius * 1.5
            elif zoom_level <= 8:
                point_radius = base_radius * 1.2
            elif zoom_level <= 9:
                point_radius = base_radius
            else:
                point_radius = base_radius * 0.8
                
            # Adjust radius if there are many points to prevent overcrowding
            if len(map_data) > 50:
                point_radius *= 0.7
        else:
            center_lat, center_lon = 37.7749, -122.4194  # Default: San Francisco
            zoom_level = 9
            point_radius = 400  # Increased default radius

        # Create the main point layer - using brighter color with higher opacity
        scatter_layer = pdk.Layer(
            'ScatterplotLayer',
            map_data,
            get_position=['lon', 'lat'],
            get_radius=point_radius,
            get_fill_color=[255, 30, 30, 220],  # Brighter red with higher opacity
            pickable=True,
            stroked=True,
            get_line_color=[0, 0, 0],
            get_line_width=2,  # Thicker border
            parameters={"depthTest": False}
        )
        
        # Add a pulsing effect for maps with few points
        layers = [scatter_layer]
        
        # If we have very few points, add a highlight layer
        if len(map_data) <= 10000:
            highlight_layer = pdk.Layer(
                'ScatterplotLayer',
                map_data,
                get_position=['lon', 'lat'],
                get_radius=point_radius * 6,  # Larger radius for the highlight
                get_fill_color=[255, 215, 0, 80],  # Subtle yellow glow
                pickable=False,
                stroked=False,
                parameters={"depthTest": False}
            )
            layers.insert(0, highlight_layer)  # Insert highlight beneath the main points

        view_state = pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=zoom_level,
            pitch=0,
            bearing=0
        )

        r = pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            map_style='mapbox://styles/mapbox/light-v9',
            tooltip={"text": "{tooltip_text}"}
        )

        st.pydeck_chart(r, use_container_width=True)
    else:
        st.write("No properties to display in this area.")

# Function to search properties by address or list number
def search_properties(properties_df, search_term):
    if not search_term:
        return properties_df
    
    # Convert search term to string for comparison
    search_term = str(search_term).lower()
    
    # Search in Address, City, and List Number
    address_match = properties_df['Address'].astype(str).str.lower().str.contains(search_term, na=False)
    city_match = properties_df['City'].astype(str).str.lower().str.contains(search_term, na=False)
    
    # Try to match list number if the search term is numeric
    list_number_match = pd.Series(False, index=properties_df.index)
    if search_term.isdigit():
        list_number_match = properties_df['List Number'].astype(str).str.contains(search_term, na=False)
    
    # Combine the matches
    return properties_df[address_match | city_match | list_number_match]

def filter_properties_by_viewport(df, viewport):
    return df[
        (df['Geo Latitude'].between(viewport['min_lat'], viewport['max_lat'])) &
        (df['Geo Longitude'].between(viewport['min_lng'], viewport['max_lng']))
    ]

def safe_int(value):
    try:
        return int(value) if pd.notna(value) else 0
    except:
        return 0

def is_valid_image(url):
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        return response.status_code == 200 and "image" in response.headers.get("Content-Type", "")
    except requests.RequestException:
        return False
    

# Function to display property listing with image
def display_property_listing(row, relevance_explanation, relevance_rank, summary):
    # Create columns for property details and photo
    listing_col, image_col = st.columns([3, 2])

    # def click_view_listing_details():
    #     st.session_state.clicked = True
    def click_view_listing_details(row_list_number):
        clicked_row = st.session_state.properties_df[st.session_state.properties_df['List Number'] == row_list_number]
        st.session_state.clicked = True
        # st.session_state.selected_listing_id = str(row['List Number'])
        st.session_state.selected_listing_id = row_list_number
        # st.session_state.selected_property_summary = summary
        st.session_state.selected_property_summary = str(clicked_row['Summary'].values[0])
        # st.session_state.selected_property_relevance = relevance_explanation
        st.session_state.selected_property_relevance = str(clicked_row['Relevance_Explanation'].values[0])
        st.switch_page("pages/property_details.py")
    
    st.button('View Listing Details Here', key=f"view_{row['List Number']}", on_click=click_view_listing_details, kwargs={'row_list_number':row['List Number']})
    # st.session_state.listing_buttons_list.append(st.button('View Listing Details Here', key=f"view_{row['List Number']}"))

    # if st.button("View Listing Details", key=f"view_{row['List Number']}"):
    if st.session_state.clicked:
        # Update session state when the button is pressed
        # st.session_state.selected_listing_id = str(row['List Number'])
        # st.session_state.selected_property_summary = summary
        # st.session_state.selected_property_relevance = relevance_explanation

        # Navigate to the details page
        st.switch_page("pages/property_details.py")

    with listing_col:
        st.markdown(f"""
        **{row['Address']}, {row['City']}, {row['State']} {row['Zip']}**
         
        **${safe_int(row['List Price']):,}** - {safe_int(row['Total Bedrooms'])} beds | {safe_int(row['Total Bathrooms'])} baths | {safe_int(row['Total Sqft'])} Sqft
        
        **LLM Reasoning:**  
        {relevance_explanation}  
        
        *{summary}*
        """)

    with image_col:
        # Display image if Photo URL exists and is not NaN
        if 'Photo URL' in row and pd.notna(row['Photo URL']) and row['Photo URL'].strip() and is_valid_image(row['Photo URL']):
            st.image(row['Photo URL'], width=400)
            st.markdown(f"<small><i>{row['caption']}</i></small>", unsafe_allow_html=True)
        else:
            st.image("https://i.imgur.com/Pj9k2Mn.png", width=200)
    
    # Add a small separator between listings
    st.markdown("---")


# Replace the display_paginated_listings function with this scrollable version
def display_scrollable_listings(properties_df, summaries, relevance_explanations, relevance_ranks, container=None, section_id="main", max_height="500px"):
    if container is None:
        container = st
        
    if properties_df is None or properties_df.empty:
        container.text("No properties to display.")
        return properties_df

    properties_df['Relevance Rank'] = properties_df['List Number'].map(relevance_ranks)
    properties_df_sorted = properties_df.sort_values(by='Relevance Rank')

    # Display the property count
    container.write(f"{len(properties_df)} properties found")
    
    # Create a scrollable container
    scroll_container = container.container()
    
    # Apply CSS to make it scrollable
    scroll_container.markdown(f"""
    <style>
        [data-testid="stVerticalBlock"] > [style*="flex-direction: column"] > [data-testid="stVerticalBlock"] {{
            max-height: {max_height};
            overflow-y: auto;
        }}
    </style>
    """, unsafe_allow_html=True)
    
    # Display all properties in the scrollable container
    with scroll_container:
        for idx in range(len(properties_df_sorted)):
            row = properties_df_sorted.iloc[idx]
            listing_id = row['List Number']
            summary = summaries.get(listing_id, 'No summary available')
            relevance_explanation = relevance_explanations.get(listing_id, 'No LLM reasoning available')
            relevance_rank = relevance_ranks.get(listing_id, 10)
            display_property_listing(row, relevance_explanation, relevance_rank, summary)

def display_existing_scrollable_listings(properties_df, summaries, relevance_explanations, relevance_ranks, container=None, section_id="main", max_height="500px"):
    if container is None:
        container = st
        
    if properties_df is None or properties_df.empty:
        container.text("No properties to display.")
        return properties_df

    # properties_df['Relevance Rank'] = properties_df['List Number'].map(relevance_ranks)
    properties_df_sorted = properties_df.sort_values(by='Relevance_Rank')

    # Display the property count
    container.write(f"{len(properties_df)} properties found")
    
    # Create a scrollable container
    scroll_container = container.container()
    
    # Apply CSS to make it scrollable
    scroll_container.markdown(f"""
    <style>
        [data-testid="stVerticalBlock"] > [style*="flex-direction: column"] > [data-testid="stVerticalBlock"] {{
            max-height: {max_height};
            overflow-y: auto;
        }}
    </style>
    """, unsafe_allow_html=True)
    
    # Display all properties in the scrollable container
    with scroll_container:
        for idx in range(len(properties_df_sorted)):
            row = properties_df_sorted.iloc[idx]
            listing_id = row['List Number']
            summary = row['Summary']
            relevance_explanation = row['Relevance_Explanation']
            relevance_rank = row['Relevance_Rank']
            display_property_listing(row, relevance_explanation, relevance_rank, summary)

# Function to display paginated listings
def display_paginated_listings(properties_df, summaries, relevance_explanations, relevance_ranks, page_size=5, container=None, section_id="main"):
    if container is None:
        container = st
        
    if properties_df is None or properties_df.empty:
        container.text("No properties to display.")
        return properties_df

    properties_df['Relevance Rank'] = properties_df['List Number'].map(relevance_ranks)
    properties_df_sorted = properties_df.sort_values(by='Relevance Rank')

    total_pages = (len(properties_df_sorted) + page_size - 1) // page_size
    
    # Keep track of pagination in session state with a unique key for each section
    pagination_key = f"page_number_{section_id}"
    if pagination_key not in st.session_state:
        st.session_state[pagination_key] = 0
    
    # Pagination controls
    col1, col2, col3 = container.columns([1, 3, 1])
    
    with col1:
        if st.session_state[pagination_key] > 0:
            if st.button("← Prev", key=f'prev_button_{section_id}_{st.session_state[pagination_key]}'):
                # Save the current expander state before rerunning
                st.session_state.expander_state = True  # Force it to stay open
                st.session_state[pagination_key] -= 1
                st.rerun()
    
    with col2:
        container.write(f"Page {st.session_state[pagination_key] + 1} of {total_pages} • {len(properties_df)} properties found")

    with col3:
        if st.session_state[pagination_key] < total_pages - 1:
            if st.button("Next →", key=f'next_button_{section_id}_{st.session_state[pagination_key]}'):
                # Save the current expander state before rerunning
                st.session_state.expander_state = True  # Force it to stay open
                st.session_state[pagination_key] += 1
                st.rerun()
    
    # Display current page properties
    start_idx = st.session_state[pagination_key] * page_size
    end_idx = min(start_idx + page_size, len(properties_df_sorted))

    for idx in range(start_idx, end_idx):
        row = properties_df_sorted.iloc[idx]
        listing_id = row['List Number']
        summary = summaries.get(listing_id, 'No summary available')  # Get the summary for the listing
        relevance_explanation = relevance_explanations.get(listing_id, 'No LLM reasoning available')
        relevance_rank = relevance_ranks.get(listing_id, 10)
        display_property_listing(row, relevance_explanation, relevance_rank, summary)
    
    # Add search bar
    container.markdown("### Search Properties")
    search_term = container.text_input("Search by address, city, or list number:", key=f'text_input_{section_id}_{st.session_state[pagination_key]}')
    
    # Use a unique key for search results for each section
    search_results_key = f"search_results_{section_id}"
    if search_results_key not in st.session_state:
        st.session_state[search_results_key] = None
    
    if container.button("Search", key=f'Search_{section_id}_{st.session_state[pagination_key]}'):
        search_results = search_properties(properties_df, search_term)
        # Reset pagination for this section
        st.session_state[pagination_key] = 0
        # Store search results for this section
        st.session_state[search_results_key] = search_results
        st.rerun()
    
    if container.button("Clear Search", key=f"Clear_Search_{section_id}_{st.session_state[pagination_key]}"):
        st.session_state[search_results_key] = None
        st.session_state[pagination_key] = 0
        st.rerun()
    
    # Return the current display data for use in the map
    return st.session_state[search_results_key] if st.session_state[search_results_key] is not None else properties_df
    
# for i, button in enumerate(st.session_state.listing_buttons_list):
#     if button:
#         # st.write(f"{i} button was clicked")
#         click_view_listing_details(i)

def user_input_changed():
    st.session_state.properties_df = None


st.subheader("Find your perfect home with Homiere")
search = st.text_input("Tell Homiere what you're looking for in a home:", placeholder="Describe anything...", on_change=user_input_changed)

st.markdown("*Optional Filters:*")
filter_cols = st.columns(5)
with filter_cols[0]:
    neighborhood_filter = st.selectbox("City", [""] + sorted(st.session_state.map_data["City"].unique()), on_change=user_input_changed)
with filter_cols[1]:
    price_filter = st.selectbox("Price", ["Any", "$0-500k", "$500k-1M", "$1M-2M", "$2M+"], on_change=user_input_changed)
with filter_cols[2]:
    sqft_filter = st.selectbox("Price Relative to City Average", ["Any", "Lower", "At", "Above"], on_change=user_input_changed)
with filter_cols[3]:
    mortgage_filter = st.selectbox("Monthly Mortgage", ["Any", "$0-1k", "$1k-5k", "$5-10k", "10k+"], on_change=user_input_changed)
with filter_cols[4]:
    type_filter = st.selectbox("Home Type", ["Any", "Single Family", "Condo", "Townhouse", "Multi-unit"], on_change=user_input_changed)

filter_cols = st.columns(4)
with filter_cols[0]:
    beds_filter = st.selectbox("Bed", ["Any", "1", "2", "3", "4+"], on_change=user_input_changed)
with filter_cols[1]:
    baths_filter = st.selectbox("Bath", ["Any", "1", "2", "3", "4+"], on_change=user_input_changed)
with filter_cols[2]:
    condition_filter = st.selectbox("Home Condition", ["Any", "Ready to move in", "Fixer", "Under Construction"], on_change=user_input_changed)
with filter_cols[3]:
    mom_filter = st.selectbox("Months on Market", ["Any", "0-6 Month", "6-12 Months", "1-2 Years", "2 years+"], on_change=user_input_changed)

data = {
    "description": search,
    "city": neighborhood_filter,
    "price": price_filter,
    "price_position_vs_city": sqft_filter,
    "home_type": type_filter,
    "bed": beds_filter,
    "bath": baths_filter,
    "mortgage": mortgage_filter,
    "home_condition": condition_filter,
    "months_on_market": mom_filter,
}

if st.session_state.properties_df is not None:
        # Display the properties in properties_df
        with st.spinner("Displaying saved search results..."):
            # Get the existing listing IDs
            listing_ids = st.session_state.properties_df['List Number'].tolist()
            
            # If we don't have the text data loaded yet, load it
            if st.session_state.text_data is None:
                load_text_data()
            
            # Display conclusion in a card-like container (using a default if not available)
            # conclusion = getattr(st.session_state.properties_df, 'conclusion', "Showing your previous search results.")
            conclusion = st.session_state.conclusion
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                <h3 style="margin-top: 0;">Analysis</h3>
                <p>{conclusion}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Create two columns for map and listings
            map_col, list_col = st.columns([1, 1])
            
            with map_col:
                st.markdown("### Map View")
                render_map(st.session_state.properties_df)
            
            with list_col:
                st.markdown("### Properties")
                
                # Try to get the summaries, relevance explanations, and ranks from session state
                # or initialize with default values if not available
                # summaries = getattr(st.session_state.properties_df, 'summaries', {})
                summaries = st.session_state.properties_df['Summary'].to_list()
                # relevance_explanations = getattr(st.session_state.properties_df, 'relevance_explanations', {})
                relevance_explanations = st.session_state.properties_df['Relevance_Explanation'].to_list()
                # relevance_ranks = getattr(st.session_state.properties_df, 'relevance_ranks', {})
                relevance_ranks = st.session_state.properties_df['Relevance_Rank'].to_list()
                
                # If summaries are empty, create default ones
                if not summaries:
                    summaries = {listing_id: 'No summary available' for listing_id in listing_ids}
                    
                # If explanations are empty, create default ones
                if not relevance_explanations:
                    relevance_explanations = {listing_id: 'No LLM reasoning available' for listing_id in listing_ids}
                    
                # If ranks are empty, create default ones
                if not relevance_ranks:
                    relevance_ranks = {listing_id: 99 for listing_id in listing_ids}
                
                # Use the scrollable function to display the properties
                display_existing_scrollable_listings(
                    st.session_state.properties_df, 
                    summaries, 
                    relevance_explanations, 
                    relevance_ranks, 
                    section_id="scrollable_view"
                )

# Replace the Submit button implementation with this to use the scrollable listings
if st.button("Submit"):
    st.session_state.search_triggered = True # Trigger this to remove map of everything

    if st.session_state.properties_df is None:
        with st.spinner("Searching for your perfect home..."):
            api_url = st.secrets["API_URL"]
            response = requests.post(api_url, json=data)

            if st.session_state.text_data is None:
                load_text_data()

            if response.status_code == 200:
                st.success("Data successfully submitted!")

                # Filter the properties based on the search criteria
                # import random
                # llm_results_list = [{'listings': {'PW-25009443': {'ID': 'PW-25009443', 'summary': 'Modern coastal home with modern amenities in sought-after Belmont Heights neighborhood.', 'relevance_explanation': "Matches 'spacious family home' and 'modern amenities'. Does not match 'coastal location'.", 'relevance_rank_score': '1'}, 'PW-22212731': {'ID': 'PW-22212731', 'summary': 'Mid century home with great place to call home in desirable area.', 'relevance_explanation': "Matches 'great place to call home' and 'mid century beauty'. Does not match 'spacious family home' or 'modern amenities'.", 'relevance_rank_score': '3'}, 'SW-24031911': {'ID': 'SW-24031911', 'summary': 'Single story home with charming location in highly sought-after area.', 'relevance_explanation': "Matches 'single story home' and 'charming location'. Does not match 'spacious family home' or 'modern amenities'.", 'relevance_rank_score': '2'}, 'PW-20066909': {'ID': 'PW-20066909', 'summary': 'Charming home in highly sought-after area with modern amenities.', 'relevance_explanation': "Matches 'charming home' and 'modern amenities'. Does not match 'spacious family home' or 'coastal location'.", 'relevance_rank_score': '1'}, 'SD-240019206': {'ID': 'SD-240019206', 'summary': '1 story home in highly sought-after Fletcher Hills neighborhood.', 'relevance_explanation': 'Does not match any criteria.', 'relevance_rank_score': '5'}, 'OC-25001900': {'summary': 'Spacious family home with modern amenities in a sought-after neighborhood in coastal Long Beach.', 'relevance_explanation': 'Matches query criteria: modern amenities, spacious family home, sought-after neighborhood. No match for "coastal".', 'relevance_rank_score': 1}}, 'conclusion': 'Based on the ranked list, I found one clear winner and several other listings that meet my query criteria.'},
                #                     {'listings': {'TR-24231146': {'summary': 'Coastal Long Beach Belmont Hei home with modern amenities', 'relevance_explanation': 'matches query criteria: spacious family home, modern amenities', 'relevance_rank_score': 1}, 'RS-25004437': {'summary': 'Mid century beauty with modern amenities', 'relevance_explanation': 'matches query criteria: spacious family home, modern amenities', 'relevance_rank_score': 1}, 'PW-24244901': {'summary': 'Charming single story home with no modern amenities', 'relevance_explanation': 'does not match query criteria: modern amenities', 'relevance_rank_score': 3}, 'OC-24169382': {'summary': 'Charming home with modern amenities in sought-after area', 'relevance_explanation': 'matches query criteria: spacious family home, modern amenities', 'relevance_rank_score': 1}, 'OC-24247152': {'summary': 'Highly sought-after 1 story home with no modern amenities', 'relevance_explanation': 'does not match query criteria: modern amenities', 'relevance_rank_score': 3}}, 'conclusion': 'several listings meet the query criteria, consider adding a specific location to narrow down the search'},
                #                     {'listings': {'OC-25001900': {'summary': 'Spacious family home with modern amenities in a sought-after neighborhood in coastal Long Beach.', 'relevance_explanation': 'Matches query criteria: modern amenities, spacious family home, sought-after neighborhood. No match for "coastal".', 'relevance_rank_score': 1}, 'OC-25001094': {'summary': 'Mid century modern home in a desirable location, perfect for a family looking for a great place to call home.', 'relevance_explanation': 'Matches query criteria: modern amenities, great place to call home. No match for "spacious family home" or "sought-after neighborhood".', 'relevance_rank_score': 2}, 'CV-24245903': {'summary': 'Charming single-story home with a rare opportunity to own a unique and exceptional property.', 'relevance_explanation': 'Matches query criteria: modern amenities, single-story home. No match for "spacious family home" or "sought-after neighborhood".', 'relevance_rank_score': 3}, 'V1-26959': {'summary': 'Charming home in a highly sought-after area of Anaheim, with modern amenities and a great location.', 'relevance_explanation': 'Matches query criteria: modern amenities, charming home, highly sought-after area. No match for "spacious family home".', 'relevance_rank_score': 4}, '00-25477589': {'summary': 'Highly sought-after 1-story home in Fletcher Hills with a unique lifestyle and modern amenities.', 'relevance_explanation': 'Matches query criteria: modern amenities, highly sought-after lifestyle. No match for "spacious family home" or "coastal".', 'relevance_rank_score': 5}}, 'conclusion': 'Multiple listings meet your criteria, including PW-25009443 and PW-20066909.'},
                #                     None
                # ]

                llm_results = None

                # Try a max of 3 times before stopping if LLM returns None
                for attempt in range(3):
                    llm_results = execute_retrieval(data, st.session_state.text_data)

                    if llm_results:
                        break

                listing_ids = []
                summaries = {}
                relevance_explanations = {}
                relevance_ranks = {}
                conclusion = "No conclusion available. Please submit a longer query"

                if llm_results:
                    listings = llm_results.get('listings', {})
                    if listings:
                        listing_ids = [listing['ID'] for listing in listings if 'ID' in listing]
                        summaries = {listing['ID']: listing.get('summary', 'No summary available') for listing in listings if 'ID' in listing}
                        relevance_explanations = {listing['ID']: listing.get('relevance_explanation', 'No LLM reasoning available') for listing in listings if 'ID' in listing}

                        relevance_ranks = {}
                        for listing in listings:
                            listing_id = listing['ID']
                            rank = listing.get('relevance_rank', '')
                            if isinstance(rank, str):  # If rank is a string, try to convert it to an int
                                try:
                                    rank = int(rank)
                                except ValueError:
                                    rank = 99  # If conversion fails, set rank to 10
                            elif rank == '':  # If rank is blank, set it to 10
                                rank = 99
                            relevance_ranks[listing_id] = rank

                        conclusion = llm_results.get("conclusion", "No conclusion available")

                        st.session_state.conclusion = conclusion

                        properties_df = st.session_state.map_data.loc[st.session_state.map_data['List Number'].isin(listing_ids)].copy()

                        # Create Series from the dictionaries
                        summary_series = pd.Series(summaries, name='Summary')
                        relevance_exp_series = pd.Series(relevance_explanations, name='Relevance_Explanation')
                        relevance_rank_series = pd.Series(relevance_ranks, name='Relevance_Rank')

                        # Convert to DataFrames with reset_index to make the keys a column
                        summary_df = summary_series.reset_index().rename(columns={'index': 'List Number'})
                        relevance_exp_df = relevance_exp_series.reset_index().rename(columns={'index': 'List Number'})
                        relevance_rank_df = relevance_rank_series.reset_index().rename(columns={'index': 'List Number'})

                        # Merge all DataFrames on the 'List Number' column
                        result = properties_df.merge(summary_df, on='List Number', how='left')
                        result = result.merge(relevance_exp_df, on='List Number', how='left')
                        result = result.merge(relevance_rank_df, on='List Number', how='left')

                        result_sorted = result.sort_values(by='Relevance_Rank')

                        
                        # st.session_state.properties_df = st.session_state.map_data.loc[st.session_state.map_data['List Number'].isin(listing_ids)].copy()
                        st.session_state.properties_df = result_sorted

                else:
                    st.error("Failed to retrieve valid results. Please refresh and resubmit your query.")

                st.markdown("## Search Results")

                # Display conclusion in a card-like container
                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                    <h3 style="margin-top: 0;">Analysis</h3>
                    <p>{conclusion}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Replace the tab-based interface with columns for side-by-side display
                if st.session_state.properties_df is not None and not st.session_state.properties_df.empty:
                    # Create two columns for map and listings
                    map_col, list_col = st.columns([1, 1])
                    
                    with map_col:
                        st.markdown("### Map View")
                        render_map(st.session_state.properties_df)
                    
                    with list_col:
                        st.markdown("### Properties")
                        # Use the new scrollable function instead of paginated
                        display_scrollable_listings(st.session_state.properties_df, summaries, relevance_explanations, relevance_ranks, section_id="scrollable_view")
                    
                else:
                    st.info("No properties match your search criteria.")

            else:
                st.error(f"Failed to send data. Status code: {response.status_code}")
    
    if "property_data" not in st.session_state:
        load_property_data()

if not st.session_state.search_triggered:
    render_map(st.session_state.map_data)

load_text_data()
load_property_data()
st.divider()