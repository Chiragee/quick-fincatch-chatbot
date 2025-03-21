import os
import json
import requests

import google.auth.transport.requests
import google.oauth2.id_token
from google.oauth2 import service_account
import google.generativeai as genai

def get_identity_token(audience):
    '''Fetches an identity token for the specified audience. Used for when we need to access other cloud run instances by setting header = {"Authorization": f"Bearer {token}"}
    Args:
        audience (str): The audience for the token. This is just the url of the service.

    Returns:    
        str: The identity token.

    Raises:
        Exception: If the token cannot be fetched.
    '''

        # If running under Streamlit, st.secrets is automatically available.
    import streamlit as st
    credentials_json = st.secrets["SERVICE_ACCOUNT_CREDENTIALS"]

    if not credentials_json:
        raise Exception("Service account credentials not found in secrets or environment variable.")
    
    credentials_info = json.loads(credentials_json)
    credentials_info['private_key'] = credentials_info['private_key'].replace('\\n', '\n') #HAVE TO REPLACE

    credentials = service_account.IDTokenCredentials.from_service_account_info(
        credentials_info,
        target_audience=audience
    )
    # Refresh to obtain a new token (this call does not require manual reauthentication)
    credentials.refresh(google.auth.transport.requests.Request())

    if credentials.token:
        return credentials.token
 
    raise Exception("Error fetching identity token.")


def call_gemini_complete(prompt: str, model_name: str = 'gemini-2.0-flash-thinking-exp') -> str:
    """
    Calls the Gemini model synchronously without any retry logic.
    It configures the client on each call using the GOOGLE_API_KEY from environment.
    """
    
    # Set up the client with the provided Google API key
    genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
    model = genai.GenerativeModel(model_name)
    
    messages = [{"role": "user", "parts": [{'text': prompt}]}]
    response = model.generate_content(messages)
    
    # Return the final text from the response
    return response.candidates[0].content.parts[-1].text


def get_context(query: str) -> dict:
    """
    Retrieves context from the Cloud Run endpoint.
    - It first gets an identity token.
    - Then it calls the get_similar_entity_and_relationships endpoint with the required parameters.
    """
    GRAPH_OUTPUT_API_URL = 'https://graph-output-api-427867203106.asia-southeast1.run.app'
    try:
        # Get identity token for authentication
        token = get_identity_token(audience=GRAPH_OUTPUT_API_URL)
        headers = {"Authorization": f"Bearer {token}"}
        
        params = {
            'project': "FINCATCH",
            'query_content': query,
            'context_window': 100000,
            'k': 50,
            'index': True,
            'start_timestamp': 1717171200, #Saturday, June 1, 2024 12:00:00 AM GMT+08:00
            'end_timestamp': 1742400000 #Thursday, March 20, 2025 12:00:00 AM GMT+08:00
        }
        # Adjust the URL as necessary for your deployment.
        url = f"{GRAPH_OUTPUT_API_URL}/get_similar_entity_and_relationships"
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        return str(response.json())
    
    except Exception as e:
        print(f"error get_context: {e}")
        return f"error get_context: {e}. In your response, mention that an error occured getting contet with the exact error message to the user."
