import streamlit as st
st.set_page_config(layout="wide")
import os
from functions import get_identity_token, call_gemini_complete, get_context

# Show title and description.
st.title("💬 Chatbot")
st.write(
    "This is a simple chatbot that uses Google's Gemini model to generate responses. "
    "Please provide your GOOGLE API key to continue."
)

# Fetch the identity token to verify connectivity with the Cloud Run endpoint.
identity_token = get_identity_token(audience='https://graph-output-api-427867203106.asia-southeast1.run.app')
if identity_token:
    st.write("Successfully obtained identity token.")
else:
    st.write("Failed to obtain identity token.")

# Ask user for their GOOGLE API Key via `st.text_input`.
google_api_key = st.text_input("GOOGLE API Key", type="password")
if not google_api_key:
    st.info("Please add your GOOGLE API key to continue.", icon="🗝️")
else:
    # Set the GOOGLE_API_KEY in the environment so that gemini_functions can use it.
    os.environ["GOOGLE_API_KEY"] = google_api_key

    # Create a session state variable to store the chat messages.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display the existing chat messages.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Create a chat input field for the user.
    if prompt := st.chat_input("What is up?"):
        # Save and display the user prompt.
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Step 1: Retrieve context from Cloud Run using the user prompt.
        context_data = get_context(prompt)

        # Step 2: Build the Gemini prompt using the user query and the retrieved context.
        answer_prompt = f'''The user has provided a query. Content has been retrieved from a graph database based on its relevance to the query. Analyze the content and provide an answer to the users query.

        - Your response must be intelligent, logical, and answer the users query fully.
        - Ensure that you cite the information id's using square brackets in your response. For example: "this is some information [1234]". This is essential for the user to be able to verify the information.
        - The user does not have access to the content retrieved from the graph database, so you must provide all relevant information in your response. i.e dont say according to [1234] the answer is X. You must actually provide the specific answer in full.
        - Your output format must be structured markdown. No preliminary comments or markdown tags are allowed, your response must directly answer the users query and be in markdown format.

        QUERY: {prompt}

        CONTENT:
        {context_data}

        - Your response must be intelligent, logical, and answer the users query fully.
        - Ensure that you cite the information id's using square brackets in your response. For example: "this is some information [1234]". This is essential for the user to be able to verify the information.
        - The user does not have access to the content retrieved from the graph database, so you must provide all relevant information in your response. i.e dont say according to [1234] the answer is X. You must actually provide the specific answer in full.
        - Your output format must be structured markdown. No preliminary comments or markdown tags are allowed, your response must directly answer the users query and be in markdown format.
        '''

        # Step 3: Query the Gemini API with the constructed prompt.
        try:
            response_text = call_gemini_complete(answer_prompt, model_name = 'gemini-2.5-pro-exp-03-25')

            response_text = response_text.strip()

            if response_text.startswith('```markdown'):
                response_text = response_text.split('```markdown', 1)[1]
            
            if response_text.endswith('```'):
                response_text = response_text.rsplit('```', 1)[0]

            response_text = response_text.strip()

            if not response_text:
                response_text = "No response from the model, please try again."

        except Exception as e:
            st.error(f"Error call_gemini_complete: {e}")
            response_text = f"Error call_gemini_complete: {e}"

        # Step 4: Display the Gemini response.
        with st.chat_message("assistant"):
            st.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})
