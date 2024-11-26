# import oci
import datetime
import functools
import os
from io import StringIO
from typing import Any, List, Mapping, Optional

import cohere
import streamlit as st
from pages.utils.style import set_page_config

set_page_config()

#-------------------------------------------------------------------
cohere_api_key = os.getenv('COHERE_API_KEY')
cohere_client = cohere.Client(api_key=cohere_api_key)

#-------------------------------------------------------------------
def timeit(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        start_time = datetime.datetime.now()
        result = func(*args, **kwargs)
        elapsed_time = datetime.datetime.now() - start_time
        print('function [{}] finished in {} ms'.format(
            func.__name__, str(elapsed_time)))
        return result
    return new_func

#-------------------------------------------------------------------
@timeit
def generate_cohere_response(prompt: str) -> str:
    """Generates a response from Cohere's large language model."""
    response = cohere_client.generate(
        model="command-xlarge-nightly",  # Use the trial model or the one assigned to you
        prompt=prompt,
        max_tokens=500,
        temperature=0.7,
    )
    return response.generations[0].text.strip()

#-------------------------------------------------------------------
# Main page setup
st.header("How can I help you today?")
st.info('Select a page on the side menu or use the chat below.', icon="📄")

# Sidebar with helpful hints
with st.sidebar:
    st.success("Hints & Commands", icon="💡")
    st.markdown(
        """
        **Commands you can use:**
        - `/repeat`: Repeats the last response.
        - `/help`: Displays this command list.
        
        **Chat Tips:**
        - Ask any question or seek guidance.
        - Use specific keywords for better responses.
        """
    )

#-------------------------------------------------------------------
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []
if "last_response" not in st.session_state:
    st.session_state.last_response = ""
if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = ""

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.divider()

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Add a spinner for loading while generating the response
    with st.spinner("Prompting to LLM..."):
        if prompt.startswith("/"):
            if prompt.split(" ")[0] == "/repeat":
                response = st.session_state.last_response
            elif prompt.split(" ")[0] == "/help":
                response = "Command list available: /repeat, /help"
            else:
                response = "Unknown command. Use /help for a list of commands."
        else:
            response = generate_cohere_response(prompt)

    # Replace the loading spinner with the actual response
    st.chat_message("assistant").markdown(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Save response and prompt
    st.session_state.last_response = response
    st.session_state.last_prompt = prompt
