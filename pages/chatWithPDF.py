import logging
import tempfile
import oracledb
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_cohere import ChatCohere
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_community.vectorstores.oraclevs import OracleVS
from langchain_community.vectorstores.utils import DistanceStrategy
from pages.utils.htmlTemplates import bot_template, css, user_template
import pages.utils.config as config
import time
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings

# Initialize Sentence Transformers model
embeddings = SentenceTransformer('all-MiniLM-L6-v2')  # You can choose any available model

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def get_pdf_text(pdf_files):
    """Extract text from PDF files."""
    text = ""
    try:
        for pdf_file in pdf_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(pdf_file.read())
                temp_file_path = temp_file.name

            loader = PyPDFLoader(temp_file_path)
            documents = loader.load()

            for doc in documents:
                text += doc.page_content
    except Exception as e:
        logger.error(f"Failed to extract text from PDFs: {e}")
        raise RuntimeError("An error occurred while processing the PDF files. Please try again.")
    return text

def get_chunk_text(text):
    """Split text into manageable chunks."""
    try:
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
    except Exception as e:
        logger.error(f"Error splitting text into chunks: {e}")
        raise RuntimeError("Failed to split text into chunks.")
    return chunks

def get_vector_store(text_chunks):
    """
    Create a vector store for document retrieval using Sentence Transformer embeddings.
    """
    try:
        logger.info("Initializing Sentence Transformer embeddings...")
        
        # Initialize Sentence Transformer model
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Choose a suitable model
        
        # Convert text chunks to Document objects
        documents = [Document(page_content=chunk) for chunk in text_chunks]
        
        # Create vector store
        logger.info("Creating vector store...")
        if config.DB_TYPE == "oracle":
            connection = oracledb.connect(
                user=config.ORACLE_USERNAME, 
                password=config.ORACLE_PASSWORD, 
                dsn=config.ORACLE_DSN
            )
            vectorstore = OracleVS.from_documents(
                documents=documents,
                embedding=embedding_model,
                client=connection,
                table_name=config.ORACLE_TABLE_NAME,
                distance_strategy=DistanceStrategy.DOT_PRODUCT
            )
        else:
            vectorstore = Qdrant.from_documents(
                documents=documents,
                embedding=embedding_model,
                location=config.QDRANT_LOCATION,
                collection_name=config.QDRANT_COLLECTION_NAME,
                distance_func=config.QDRANT_DISTANCE_FUNC
            )
        logger.info("Vector store creation successful.")
    except oracledb.DatabaseError as db_error:
        logger.error(f"Database error: {db_error}")
        raise RuntimeError("Failed to connect to the Oracle database.")
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        raise RuntimeError("An error occurred while setting up the vector store.")
    return vectorstore

def get_conversation_chain(vector_store):
    """Create a conversation chain for user interaction."""
    try:
        llm = ChatCohere(model="command-r-plus-08-2024")
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory
        )
    except Exception as e:
        logger.error(f"Failed to initialize conversation chain: {e}")
        raise RuntimeError("An error occurred while setting up the conversation chain.")
    return conversation_chain

def handle_user_input(question):
    """Handle user input and update chat history."""
    try:
        response = st.session_state.conversation({'question': question})
        st.session_state.chat_history = response['chat_history']
    except Exception as e:
        logger.error(f"Error handling user input: {e}")
        st.error("Failed to process your query. Please try again.")

def main():
    """Main application entry point."""
    st.set_page_config(page_title='Chat with Your own PDFs', page_icon=':books:')
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    def handle_submit():
        """Callback for text input submission."""
        if st.session_state.user_input and st.session_state.conversation:
            handle_user_input(st.session_state.user_input)
            st.session_state.user_input = ""

    st.header('Chat with Your own PDFs :books:')

    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.markdown(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.markdown(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

    input_container = st.container()
    with input_container:
        st.text_input(
            "Ask anything to your PDF:",
            key="user_input",
            on_change=handle_submit
        )

    with st.sidebar:
        st.subheader("Upload your Documents Here:")
        pdf_files = st.file_uploader("Choose your PDF Files", type=['pdf'], accept_multiple_files=True)

        if st.button("OK", disabled=not pdf_files or st.session_state.is_processing):
            st.session_state.is_processing = True
            with st.spinner("Processing your PDFs..."):
                try:
                    raw_text = get_pdf_text(pdf_files)
                    text_chunks = get_chunk_text(raw_text)
                    vector_store = get_vector_store(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vector_store)
                    st.success("Processing complete! You can now start asking questions.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    logger.error(f"Error processing PDFs: {e}")
                finally:
                    st.session_state.is_processing = False

if __name__ == '__main__':
    main()
