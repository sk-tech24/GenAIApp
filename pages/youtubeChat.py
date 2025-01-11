import datetime
import functools
import re
import langchain
import oracledb
import pages.utils.config as config  # Import the configuration
import streamlit as st
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain_cohere import ChatCohere
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
# New imports
from langchain_community.embeddings import CohereEmbeddings, OCIGenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_community.vectorstores.oraclevs import OracleVS
from langchain_community.vectorstores.utils import DistanceStrategy
from pages.utils.style import set_page_config
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

set_page_config()

langchain.verbose = False

def timeit(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        start_time = datetime.datetime.now()
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            print(f"Error occurred in function {func.__name__}: {e}")
            return None
        finally:
            elapsed_time = datetime.datetime.now() - start_time
            print('function [{}] finished in {} ms'.format(func.__name__, str(elapsed_time)))
    return new_func

def handle_exception(func):
    """Wrapper to catch all exceptions and log user-friendly messages."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error in function {func.__name__}: {e}")
            st.error(f"An error occurred in {func.__name__}. Please try again later.")
            return None
    return wrapper

@timeit
@handle_exception
def fetching_youtubeid(youtubeid):
    if "youtu" in youtubeid:
        data = re.findall(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", youtubeid)
        if data:
            return data[0]
        else:
            raise ValueError("Invalid YouTube ID/URL format.")
    return youtubeid

@timeit
@st.cache_resource(show_spinner="Fetching data from Youtube...")
@handle_exception
def fetching_transcript(youtubeid, chunk_size, chunk_overlap):
    youtubeid = fetching_youtubeid(youtubeid)
    
    try:
        transcript = YouTubeTranscriptApi.get_transcript(youtubeid, languages=['pt', 'en'])
    except Exception as e:
        print(f"Error fetching transcript: {e}")
        st.error("Failed to fetch YouTube transcript. Please check the video ID or try again later.")
        return None
    
    formatter = TextFormatter()
    text = formatter.format_transcript(transcript)

    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Use OCIGenAIEmbeddings for embeddings
    embeddings = CohereEmbeddings(
        model=config.EMBEDDING_MODEL,  # Replace with your Cohere model name (e.g., "embed-english-v2.0")
        user_agent="langchain"
    )

    if config.DB_TYPE == "oracle":
        try:
            connection = oracledb.connect(user=config.ORACLE_USERNAME, password=config.ORACLE_PASSWORD, dsn=config.ORACLE_DSN)
            print("Connection to OracleDB successful!")
        except oracledb.DatabaseError as db_err:
            print(f"Oracle connection error: {db_err}")
            st.error("Failed to connect to OracleDB. Please check the database connection or configuration.")
            return None
        except Exception as e:
            print(f"Error connecting to OracleDB: {e}")
            st.error("An error occurred while connecting to OracleDB. Please try again later.")
            return None
        
        if connection:
            try:
                knowledge_base = OracleVS.from_texts(
                    chunks,
                    embeddings,
                    client=connection,
                    table_name=config.ORACLE_TABLE_NAME,
                    distance_strategy=DistanceStrategy.DOT_PRODUCT,
                )
            except Exception as e:
                print(f"Error creating knowledge base from Oracle: {e}")
                st.error("Failed to create knowledge base from OracleDB. Please check the configuration.")
                return None
        else:
            raise Exception("Failed to connect to OracleDB.")
    else:
        try:
            knowledge_base = Qdrant.from_texts(
                chunks,
                embeddings,
                location=config.QDRANT_LOCATION,
                collection_name=config.QDRANT_COLLECTION_NAME,
                distance_func=config.QDRANT_DISTANCE_FUNC
            )
        except Exception as e:
            print(f"Error creating knowledge base from Qdrant: {e}")
            st.error("Failed to create knowledge base using Qdrant. Please check the configuration.")
            return None

    return knowledge_base

@timeit
@handle_exception
def prompting_llm(user_question, _knowledge_base, _chain, k_value):
    try:
        with st.spinner(text="Prompting LLM..."):
            doc_to_prompt = _knowledge_base.similarity_search(user_question, k=k_value)
            docs_stats = _knowledge_base.similarity_search_with_score(user_question, k=k_value)
            print('\n# ' + datetime.datetime.now().astimezone().isoformat() + ' =====================================================')
            print("Prompt: " + user_question + "\n")
            for x in range(len(docs_stats)):
                try:
                    print('# ' + str(x) + ' -------------------')
                    content, score = docs_stats[x]
                    print("Content: " + content.page_content)
                    print("\nScore: " + str(score) + "\n")
                except Exception as e:
                    print(f"Error processing document {x}: {e}")
                    continue

            prompt_len = _chain.prompt_length(docs=doc_to_prompt, question=user_question)
            st.write(f"Prompt len: {prompt_len}")

            response = _chain.invoke({"input_documents": doc_to_prompt, "question": user_question}, return_only_outputs=True).get("output_text")
            print("-------------------\nResponse:\n" + response + "\n")
            return response
    except Exception as e:
        print(f"Error in prompting LLM: {e}")
        st.error("An error occurred while prompting the LLM. Please try again later.")
        return None

@timeit
@handle_exception
def chunk_search(user_question, _knowledge_base, k_value):
    try:
        with st.spinner(text="Prompting LLM..."):
            doc_to_prompt = _knowledge_base.similarity_search(user_question, k=k_value)
            docs_stats = _knowledge_base.similarity_search_with_score(user_question, k=k_value)
            result = '  \n ' + datetime.datetime.now().astimezone().isoformat()
            result = result + "  \nPrompt: " + user_question + "  \n"
            for x in range(len(docs_stats)):
                try:
                    result = result + '  \n' + str(x) + ' -------------------'
                    content, score = docs_stats[x]
                    result = result + "  \nContent: " + content.page_content
                    result = result + "  \n  \nScore: " + str(score) + "  \n"
                except Exception as e:
                    print(f"Error processing document {x}: {e}")
                    continue
            return result
    except Exception as e:
        print(f"Error in chunk search: {e}")
        st.error("An error occurred while performing chunk search. Please try again later.")
        return None

def parseYoutubeURL(url: str):
    try:
        data = re.findall(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
        if data:
            return data[0]
        return ""
    except Exception as e:
        print(f"Error parsing YouTube URL: {e}")
        st.error("An error occurred while parsing the YouTube URL. Please try again.")
        return ""

def main():
    llm = ChatCohere(model="command-r-plus-08-2024")
    chain = load_qa_chain(llm, chain_type="stuff")

    if hasattr(chain.llm_chain.prompt, 'messages'):
        for message in chain.llm_chain.prompt.messages:
            if hasattr(message, 'template'):
                message.template = message.template.replace("Helpful Answer:", "\n### Assistant:")

    # st.set_page_config(page_title="Ask Youtube Video", layout="wide")
    st.header("Ask Youtube using GEN AI")
    youtubeid = st.text_input('Add the desired Youtube video ID or URL here.')

    with st.expander("Advanced options"):
        k_value = st.slider('Top K search | default = 6', 2, 10, 6)
        chunk_size = st.slider('Chunk size | default = 1000 [Rebuilds the Vector store]', 500, 1500, 1000, step=20)
        chunk_overlap = st.slider('Chunk overlap | default = 20 [Rebuilds the Vector store]', 0, 400, 200, step=20)
        chunk_display = st.checkbox('Show chunk text', False)

    if youtubeid:
        # Check if it's a valid youtube link
        youtubeid = fetching_youtubeid(youtubeid)
        if not youtubeid:
            st.error("Invalid YouTube ID")
        else:
            knowledge_base = fetching_transcript(youtubeid, chunk_size, chunk_overlap)
            if knowledge_base:
                user_question = st.text_area("What do you want to ask this video?")
                if user_question:
                    result = prompting_llm(user_question, knowledge_base, chain, k_value)
                    if result:
                        st.write(result)

#-------------------------------------------------------------------
if __name__ == "__main__":
    main()