import datetime
import functools
import logging
import os
import re
import getpass

import langchain
import oracledb
import streamlit as st
from bs4 import BeautifulSoup
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain_cohere import ChatCohere
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.vectorstores import Qdrant
from langchain_community.vectorstores.oraclevs import OracleVS
from langchain_community.vectorstores.utils import DistanceStrategy
from pages.utils.style import set_page_config
from playwright.sync_api import sync_playwright
import pages.utils.config as config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

set_page_config()

# Ensure Playwright is installed
os.system("playwright install")

# Turn off langchain verbosity
langchain.verbose = False

#-------------------------------------------------------------------
def get_cohere_api_key():
    if not os.getenv("COHERE_API_KEY"):
        os.environ["COHERE_API_KEY"] = getpass.getpass("Enter your Cohere API key: ")
    return os.getenv('COHERE_API_KEY')

cohere_api_key = get_cohere_api_key()

#-------------------------------------------------------------------
def timeit(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        start_time = datetime.datetime.now()
        result = func(*args, **kwargs)
        elapsed_time = datetime.datetime.now() - start_time
        logger.info(f'Function [{func.__name__}] finished in {elapsed_time} ms')
        return result
    return new_func

#-------------------------------------------------------------------
@timeit
@st.cache_data(show_spinner="Fetching data from Wikipedia...")
def fetching_article(wikipediatopic, chunk_size, chunk_overlap):
    embeddings = CohereEmbeddings(
        model=config.EMBEDDING_MODEL,
        user_agent="langchain"
    )

    wikipage = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    text = wikipage.run(wikipediatopic)

    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    chunks = text_splitter.split_text(text)

    if config.DB_TYPE == "oracle":
        try:
            connection = oracledb.connect(user=config.ORACLE_USERNAME, password=config.ORACLE_PASSWORD, dsn=config.ORACLE_DSN)
            logger.info("Connection to OracleDB successful!")
        except Exception as e:
            logger.error(f"Connection to OracleDB failed: {e}")
            connection = None

        if connection:
            knowledge_base = OracleVS.from_texts(
                chunks,
                embeddings,
                client=connection,
                table_name=config.ORACLE_TABLE_NAME,
                distance_strategy=DistanceStrategy.DOT_PRODUCT,
            )
        else:
            raise Exception("Failed to connect to OracleDB.")
    else:
        knowledge_base = Qdrant.from_texts(
            chunks,
            embeddings,
            location=config.QDRANT_LOCATION,
            collection_name=config.QDRANT_COLLECTION_NAME,
            distance_func=config.QDRANT_DISTANCE_FUNC
        )
    return knowledge_base

#-------------------------------------------------------------------
def fetch_dynamic_page_with_playwright(url: str) -> str:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.route("**/*", lambda route: route.abort() if route.request.resource_type in ["image", "stylesheet", "font"] else route.continue_())

        page.goto(url, timeout=60000, wait_until="domcontentloaded")
        content = page.content()
        browser.close()
    return content

#-------------------------------------------------------------------
@timeit
@st.cache_resource(show_spinner="Fetching data from URL...")
def fetching_url(userinputquery, chunk_size, chunk_overlap):
    embeddings = CohereEmbeddings(
        model=config.EMBEDDING_MODEL,
        user_agent="langchain"
    )

    try:
        html_content = fetch_dynamic_page_with_playwright(userinputquery)
        soup = BeautifulSoup(html_content, 'html.parser')
        text = soup.get_text()
    except Exception as e:
        st.error(f"Error fetching webpage: {e}")
        return None

    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    chunks = text_splitter.split_text(text)

    if config.DB_TYPE == "oracle":
        try:
            connection = oracledb.connect(user=config.ORACLE_USERNAME, password=config.ORACLE_PASSWORD, dsn=config.ORACLE_DSN)
            logger.info("Connection to OracleDB successful!")
        except Exception as e:
            logger.error(f"Connection to OracleDB failed: {e}")
            connection = None

        if connection:
            knowledge_base = OracleVS.from_texts(
                chunks,
                embeddings,
                client=connection,
                table_name=config.ORACLE_TABLE_NAME,
                distance_strategy=DistanceStrategy.DOT_PRODUCT,
            )
        else:
            raise Exception("Failed to connect to OracleDB.")
    else:
        knowledge_base = Qdrant.from_texts(
            chunks,
            embeddings,
            location=config.QDRANT_LOCATION,
            collection_name=config.QDRANT_COLLECTION_NAME,
            distance_func=config.QDRANT_DISTANCE_FUNC
        )
    return knowledge_base

#-------------------------------------------------------------------
@timeit
def prompting_llm(user_question, _knowledge_base, _chain, k_value):
    with st.spinner(text="Prompting LLM..."):
        doc_to_prompt = _knowledge_base.similarity_search(user_question, k=k_value)
        docs_stats = _knowledge_base.similarity_search_with_score(user_question, k=k_value)
        logger.info(f"Prompt: {user_question}")
        for x in range(len(docs_stats)):
            try:
                logger.info(f"Chunk {x} -------------------")
                content, score = docs_stats[x]
                logger.info(f"Content: {content.page_content}")
                logger.info(f"Score: {score}")
            except Exception as e:
                logger.error(f"Error processing chunk {x}: {e}")

        prompt_len = _chain.prompt_length(docs=doc_to_prompt, question=user_question)
        st.write(f"Prompt len: {prompt_len}")

        response = _chain.invoke({"input_documents": doc_to_prompt, "question": user_question}, return_only_outputs=True).get("output_text")
        logger.info(f"Response: {response}")
        return response

#-------------------------------------------------------------------
@timeit
def chunk_search(user_question, _knowledge_base, k_value):
    with st.spinner(text="Prompting LLM..."):
        doc_to_prompt = _knowledge_base.similarity_search(user_question, k=k_value)
        docs_stats = _knowledge_base.similarity_search_with_score(user_question, k=k_value)
        result = f"{datetime.datetime.now().astimezone().isoformat()}  \nPrompt: {user_question}  \n"
        for x in range(len(docs_stats)):
            try:
                result += f"  \n{x} -------------------"
                content, score = docs_stats[x]
                result += f"  \nContent: {content.page_content}"
                result += f"  \n  \nScore: {score}  \n"
            except Exception as e:
                logger.error(f"Error processing chunk {x}: {e}")
        return result

#-------------------------------------------------------------------
def main():
    llm = ChatCohere(model="command-r-plus-08-2024")
    chain = load_qa_chain(llm, chain_type="stuff")

    if hasattr(chain.llm_chain.prompt, 'messages'):
        for message in chain.llm_chain.prompt.messages:
            if hasattr(message, 'template'):
                message.template = message.template.replace("Helpful Answer:", "\n### Assistant:")

    st.header("Ask any website using Gen AI")
    userinputquery = st.text_input('Add the desired Wikipedia topic here, or a Website URL')

    with st.expander("Advanced options"):
        k_value = st.slider('Top K search | default = 5', 2, 10, 5)
        chunk_size = st.slider('Chunk size | default = 1000 [Rebuilds the Vector store]', 500, 1500, 1000, step=20)
        chunk_overlap = st.slider('Chunk overlap | default = 20 [Rebuilds the Vector store]', 0, 400, 200, step=20)
        chunk_display = st.checkbox("Display chunk results")

    if userinputquery:
        if validate_input(userinputquery):
            if userinputquery.startswith("http"):
                knowledge_base = fetching_url(userinputquery, chunk_size, chunk_overlap)
            else:
                knowledge_base = fetching_article(userinputquery, chunk_size, chunk_overlap)

            user_question = st.text_input("Ask a question about the loaded content:")

            promptoption = st.selectbox(
                '...or select a prompt templates',
                ("Summarize the page", "Summarize the page in bullet points"), index=None,
                placeholder="Select a prompt template..."
            )

            if promptoption:
                user_question = promptoption

            if user_question:
                response = prompting_llm("This is a web page, based on this text " + user_question.strip(), knowledge_base, chain, k_value)
                st.write("_"+user_question.strip()+"_")
                st.write(response)
                if chunk_display:
                    chunk_display_result = chunk_search(user_question.strip(), knowledge_base, k_value)
                    with st.expander("Chunk results"):
                        st.code(chunk_display_result)
        else:
            st.error("Invalid input. Please enter a valid Wikipedia topic or a URL starting with 'http'.")

#-------------------------------------------------------------------
def validate_input(userinputquery):
    url_pattern = re.compile(r'^https?://')
    wiki_pattern = re.compile(r'^[^/]+$')
    if url_pattern.match(userinputquery) or wiki_pattern.match(userinputquery):
        return True
    return False

#-------------------------------------------------------------------
if __name__ == "__main__":
    main()
