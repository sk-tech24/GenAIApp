import os
import datetime
import functools
import logging
import getpass
import cohere
import streamlit as st
from pages.utils.style import set_page_config

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

set_page_config()

# -------------------------------------------------------------------

def get_cohere_api_key():
    """Fetch the Cohere API key from environment or prompt the user."""
    if not os.getenv("COHERE_API_KEY"):
        os.environ["COHERE_API_KEY"] = getpass.getpass("Enter your Cohere API key: ")
    return os.getenv("COHERE_API_KEY")

cohere_api_key = get_cohere_api_key()
try:
    cohere_client = cohere.ClientV2(api_key=cohere_api_key)
except Exception as e:
    logger.error(f"Failed to initialize Cohere client: {e}")
    st.error("An error occurred while initializing the AI service. Please check your API key and try again.")
    raise

# -------------------------------------------------------------------

def timeit(func):
    """Decorator to log the execution time of functions."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.datetime.now()
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            raise
        finally:
            elapsed_time = datetime.datetime.now() - start_time
            logger.info(f"Function [{func.__name__}] executed in {elapsed_time}")
    return wrapper

# -------------------------------------------------------------------

@timeit
def generate_cohere_response(prompt):
    """Send the user's prompt to Cohere and return the response."""
    try:
        logger.info(f"Sending request to Cohere with prompt: {prompt}")
        response = cohere_client.chat(
            model="command-r-plus-08-2024",
            messages=[{"role": "user", "content": prompt}]
        )
        logger.info(f"Response received: {response}")
        return response.message.content[0].text
    except cohere.error.CohereError as e:
        logger.error(f"Cohere API error: {e}")
        raise ValueError("An error occurred while communicating with the AI service. Please try again later.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise ValueError("Something went wrong. Please try again.")

# -------------------------------------------------------------------

# Main page setup
st.header("How can I help you today?")
st.info("Select a page on the side menu or use the chat below.", icon="📄")

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

# -------------------------------------------------------------------

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_response" not in st.session_state:
    st.session_state.last_response = ""

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.divider()

# React to user input
if prompt := st.chat_input("What is up?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Add a spinner for loading while generating the response
    with st.spinner("Generating a thoughtful response..."):
        if prompt.startswith("/"):
            command = prompt.split(" ")[0]
            if command == "/repeat":
                response = st.session_state.last_response or "No response to repeat yet!"
            elif command == "/help":
                response = "Commands: `/repeat` - repeats the last response, `/help` - lists available commands."
            else:
                response = "Unknown command. Use `/help` to see the list of commands."
        else:
            try:
                response = generate_cohere_response(prompt)
            except ValueError as e:
                response = str(e)
            except Exception as e:
                response = "An unexpected error occurred. Please try again later."
                logger.error(f"Failed to generate response: {e}")

    # Display the response
    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.last_response = response
