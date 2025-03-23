"""
NOTE Items:
this script should be imported into main.py and called with the required parameters
main.py should manage packages using:
    pip install -r '../requirements.txt'
expected usage in main.py:
    from retrieval_pipeline import execute_retrieval
    final_results = execute_retrieval(input_dict, processed_text_data)

TODO Items: refine and update the following retrieval parameters:
    index_name
    top_k
    llm_model
    listing_truncation_length_chars
    max_output_tokens
    generation_length_words
    ...and content of prompt template
"""

import warnings
import pandas as pd
import pinecone
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import boto3
import openai
from openai import OpenAI
import streamlit as st

# apply preferred pandas settings for printing and debugging
pd.set_option('display.max_columns', None)  # show all cols
pd.set_option('display.max_rows', None)  # show all rows
pd.set_option('display.expand_frame_repr', False)  # disable line wrapping
pd.set_option('display.max_colwidth', None)  # show full content of each column
# apply preferred warnings settings for printing and debugging
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# define a translation function between front-end dict and structured pinecone syntax
def translate_filters(user_filters):
    pinecone_filters = {} # dict to hold pinecone-compatible filters
    price_map = {"$0-500k": (0, 500000), "$500k-1M": (500000, 1000000), "$1M-2M": (1000000, 2000000), "$2M+": (2000000, None)} # map price ranges to numeric ranges
    # price filter
    if user_filters['list_price'] != "Any":
        price_range = price_map.get(user_filters['list_price'])
        if price_range:
            lower, upper = price_range
            if lower:
                pinecone_filters['list_price'] = {"$gte": lower}
            if upper:
                pinecone_filters['list_price'] = {"$lt": upper}
    monthly_mortgage_map = {"$0-1k": (0, 1000), "$1k-5k": (1000, 5000), "$5-10k": (5000, 10000), "10k+": (10000, None)} # map monthly mortgages to numeric ranges
    # price filter
    if user_filters['monthly_mortgage'] != "Any":
        price_range = monthly_mortgage_map.get(user_filters['monthly_mortgage'])
        if price_range:
            lower, upper = price_range
            if lower:
                pinecone_filters['monthly_mortgage'] = {"$gte": lower}
            if upper:
                pinecone_filters['monthly_mortgage'] = {"$lt": upper}
    months_on_market_map = {"0-6 Month": (0, 6), "6-12 Months": (6, 12), "1-2 Years": (12, 24), "2 years+": (24, None)} # map months on market to numeric ranges
    # months on market filter
    if user_filters['months_on_market'] != "Any":
        price_range = months_on_market_map.get(user_filters['months_on_market'])
        if price_range:
            lower, upper = price_range
            if lower:
                pinecone_filters['months_on_market'] = {"$gte": lower}
            if upper:
                pinecone_filters['months_on_market'] = {"$lt": upper}
    # map bedrooms and bathrooms
    def parse_room_filter(value, is_exact=True):
        if value == "4+":
            return 4, None  # no upper limit
        elif value == "Any":
            return None
        else:
            return int(value), (int(value) + 1) if is_exact else None # assume there's always no upper limit
    # bedrooms and bathrooms filters
    bedrooms = parse_room_filter(user_filters['total_bedrooms'], 
                                 is_exact=(user_filters['total_bedrooms'] not in ["4+", "Any"]))
    if bedrooms:
        if bedrooms[1]:
            pinecone_filters['total_bedrooms'] = {"$gte": bedrooms[0], "$lt": bedrooms[1]}
        else:
            # Range match: >= lower only
            pinecone_filters['total_bedrooms'] = {"$gte": bedrooms[0]}
    bathrooms = parse_room_filter(user_filters['total_bathrooms'],
                                  is_exact=(user_filters['total_bathrooms'] not in ["4+", "Any"]))
    if bathrooms:
        if bathrooms[1]:
            pinecone_filters['total_bathrooms'] = {"$gte":bathrooms[0], "$lt": bathrooms[1]}
        else:
            # Range match: >= lower only
            pinecone_filters['total_bathrooms'] = {"$gte": bathrooms[0]}
    # additional filters 'city', 'home_type', 'property_condition' can be directly mapped if not 'Any'
    if user_filters['city'] not in ["Any", ""]:
        pinecone_filters['city'] = user_filters['city'].title()
    if user_filters['type'] != "Any":
        pinecone_filters['type'] = user_filters['type']
    if user_filters['property_condition'] != "Any":
        pinecone_filters['property_condition'] = user_filters['property_condition']
    return pinecone_filters

# define a function to load OpenAI API key from a file
def load_openai_api_key():
    openai.api_key = st.secrets["OPENAI_API_KEY"]  # Set the API key globally
    return openai.api_key

# define a function to vectorize text using OpenAI embedding model
def get_openai_embedding(text):
    # initialize API clients
    api_key = load_openai_api_key()
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return response.data[0].embedding
    
# define a function to initialize piinecone vdb
def initialize_pinecone(api_key_path=st.secrets["PINECONE_API_KEY"]):
    # with open(api_key_path, 'r') as file:
    #     api_key = st.secrets["PINECONE_API_KEY"]
    api_key = st.secrets["PINECONE_API_KEY"]
    pc = Pinecone(api_key=api_key)
    return pc

# define a function to check or create index in the vdb with ServerlessSpec
def get_or_create_index(pc, index_name="test4", vector_dim=3072):
    if index_name not in pc.list_indexes().names():
        pc.create_index(name=index_name, dimension=vector_dim, metric='cosine', spec=ServerlessSpec(cloud="aws", region="us-east-1")) # anything other that us-east-1 requires paid plan
    return pc.Index(index_name)

# define a function to retrieve records in a pinecone index, given a query
def query_pinecone(pc, query, filter_query, top_k, index_name='test4', vector_dim=3072):
    # dict mapping the index names to model names and their vector dimensions
    index_to_model = {
        'test1': ('all-mpnet-base-v2', 768),
        'test2': ('all-distilroberta-v1', 768),
        'test3': ('all-MiniLM-L6-v2', 384),
        'test4': ('text-embedding-3-large', 3072)
    }
    model_name, vector_dim = index_to_model.get(index_name, ("Unknown Model", 768))
    # generate query embeddings based on model
    if model_name == 'text-embedding-3-large':
        query_embedding = get_openai_embedding(query)
    else:
        query_embedding = SentenceTransformer(model_name).encode(query).tolist()
    pc_index = get_or_create_index(pc, index_name=index_name, vector_dim=vector_dim)
    top_results = pc_index.query(vector=query_embedding, top_k=top_k, filter=filter_query)
    unranked_retrievals = [{"id": match["id"], "score": match["score"]} for match in top_results['matches']]
    return unranked_retrievals

# define a function to generate the full prompt for all tasks
def generate_prompt(unranked_retrievals, user_query, listing_truncation_length_chars=50, generation_length_words=30):
    N = len(unranked_retrievals)
    prompt = f"""
    You are an exceptional Assistant helping with a home search.  Below is a user query that describes what I'm looking for, along with a list of {N} home listings.  Each listing  has an ID and a Description.  Your task is to process these listings according to the Tasks specified below, without using any programming code.
    
    User Query: {user_query}
    """
    N = len(unranked_retrievals)
    for idx, listing in enumerate(unranked_retrievals):
        prompt += f"\n\nID: {listing['id']}"
        prompt += f"\nDescription: {listing['enriched_text'][:listing_truncation_length_chars]}"
    prompt += f"""

    The following Analysis Tasks should be performed for each listing:
    Analysis Task 1: Provide a brief 1-sentence summary of the listing's Description (not exceeding {generation_length_words} words).
    Analysis Task 2: Think step by step to assess the relevance of the listing to the query, based on the Description.  You should note which query criteria match, which critera differ, and which criteria are not addressed.
    Analysis Task 3: Based on the relevance assessment in Analysis Task 2, create a brief 1-sentence explanation of how well the listing meets the query's criteria (not exceeding {generation_length_words} words).

    The following Synthesis Tasks should be performed once all Analysis Tasks have been completed for each listing:
    Synthesis Task 1: Initially, assign a provisional relevance rank from 1 (most relevant) to {N} (least relevant) based on your assessment. Ensure that in this initial assignment, no two listings receive the same rank.  In the case of a tie, assign ranks randomly to ensure each listing has a unique rank.
    Synthesis Task 2: After all listings have been assigned a provisional rank, carefully review all initial rankings to see if the same rank is assigned to more than one listing.  If the same rank is seen more than once, adjust the rankings based on further analysis of relevance to ensure each listing has a unique rank.  In the case of a tie, assign ranks randomly to ensure each listing has a unique rank.
    Synthesis Task 3: Provide a 1-sentence final conclusion summarizing if any listings particularly stand out, or if further information is needed to refine the search (not exceeding {generation_length_words} words).  The conclusion should not explicitly mention any Listing ID's.
    Synthesis Task 4: Return the results in valid JSON format between the markers :::BEGIN FINAL RESPONSE::: and :::END FINAL RESPONSE:::. Ensure that no instructions or additional text is included within these markers.  The JSON should have 2 top-level keys:  "listings" and "conclusion", and "listings" must be a list of length {N} where each item includes the keys "ID", "summary", "relevance_explanation", and "relevance_rank".  The results for all listings should be embedded in a single JSON structure with the following format:
    :::BEGIN FINAL RESPONSE:::
    {{
    "listings": [[
            {{
            "ID": "<the ID, as a string, of the listing>",
            "summary": "<the listing's summary, obtained in Analysis Task 1>',
            "relevance_explanation": "<the explanation of listing's relevance to my query, obtained in Analysis Task 3>",
            "relevance_rank": "<the listing's final ranking score, obtained in Synthesis Task 3>"
            }},
        ]],
    "conclusion":  "<your final conclusion, obtained in Synthesis Task 3>"
    }}
    :::END FINAL RESPONSE:::
    """
    return prompt

# define a function to initialize Bedrock client
def initialize_bedrock_client():
    return boto3.client('bedrock-runtime', 
                        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID_BEDROCK"],
                        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY_BEDROCK"],
                        region_name=st.secrets["REGION_NAME"])

# define a function to call a specified llm model
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

# define a function to parse the LLM response and store the results into a dict
def parse_final_response(llm_response):
    # define the markers
    start_marker = ":::BEGIN FINAL RESPONSE:::"
    end_marker = ":::END FINAL RESPONSE:::"
    # secondary markers to be used if primary markers aren't found
    alt_start_marker = "```"
    alt_end_marker = "```"
    # Find the last start and end of the JSON section
    start_idx = llm_response.rfind(start_marker) + len(start_marker)
    end_idx = llm_response.rfind(end_marker)
    # verify that the markers are found and in the correct order
    if start_idx == len(start_marker) - 1 or end_idx == -1:
        start_idx = llm_response.find(alt_start_marker) + len(alt_start_marker)
        end_idx = llm_response.rfind(alt_end_marker)
        if start_idx == len(alt_start_marker) - 1 or end_idx == -1:
            return None
    # extract the JSON string between the markers
    json_str = llm_response[start_idx:end_idx].strip()
    # check if the last character before the end marker is not a closing brace and append one if necessary
    if json_str[-1] != '}':
        json_str += '}'
    # convert JSON string to a Python dictionary
    try:
        final_results = json.loads(json_str)
        return final_results
    except json.JSONDecodeError as e:
        return None

# define the main function to execute retrieval
def execute_retrieval(input_dict, processed_text_data, index_name='test4', top_k=5, llm_model='llama-3-3', listing_truncation_length_chars=10000, max_output_tokens=4096, generation_length_words=50):
    #received_body = input_dict['body']['received_body'] # extract the received body
    received_body = input_dict
    user_query = received_body['description'] # extract the description
    user_filters = { # extract the metadata filters
        "city": received_body['city'],
        "list_price": received_body['price'],
        "type": received_body['home_type'],
        "total_bedrooms": received_body['bed'],
        "total_bathrooms": received_body['bath'],
        "monthly_mortgage": received_body['mortgage'],
        "property_condition": received_body['home_condition'],
        "months_on_market": received_body['months_on_market']
    }
    pinecone_filters = translate_filters(user_filters)
    pc = initialize_pinecone()
    unranked_retrievals = query_pinecone(pc, user_query, pinecone_filters, top_k=top_k, index_name=index_name)
    list_number_to_text = dict(zip(processed_text_data['List Number'], processed_text_data['enriched_text']))  # make a dict to support faster printing of listing descriptions
    for match in unranked_retrievals: # add enriched_text for each results in unranked_retrievals
        match['enriched_text'] = list_number_to_text.get(str(match['id']), "No description available")
    prompt = generate_prompt(unranked_retrievals, user_query, listing_truncation_length_chars=listing_truncation_length_chars, generation_length_words=generation_length_words)
    llm_response = call_llm(prompt, llm_model=llm_model, max_output_tokens=max_output_tokens)
    final_results = parse_final_response(llm_response)
    return final_results
