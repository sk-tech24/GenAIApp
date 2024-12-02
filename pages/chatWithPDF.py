import oracledb
import pages.utils.config as config
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
import tempfile  # Import for creating temporary files


# Function to extract text from PDF files using LangChain
def get_pdf_text(pdf_files):
    text = ""
    for pdf_file in pdf_files:
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_file.read())
            temp_file_path = temp_file.name

        # Use PyPDFLoader with the temporary file path
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()

        # Concatenate text from all pages
        for doc in documents:
            text += doc.page_content
    return text


# Function to split text into chunks
def get_chunk_text(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


# Function to create a vector store
def get_vector_store(text_chunks):
    embeddings = CohereEmbeddings(
        model=config.EMBEDDING_MODEL,
        user_agent="langchain"
    )

    documents = [Document(page_content=chunk) for chunk in text_chunks]

    if config.DB_TYPE == "oracle":
        try:
            connection = oracledb.connect(user=config.ORACLE_USERNAME, password=config.ORACLE_PASSWORD, dsn=config.ORACLE_DSN)
            print("Connection to OracleDB successful!")
        except Exception as e:
            print("Connection to OracleDB failed!")
            connection = None

        vectorstore = OracleVS.from_documents(
            documents=documents,
            embedding=embeddings,
            client=connection,
            table_name=config.ORACLE_TABLE_NAME,
            distance_strategy=DistanceStrategy.DOT_PRODUCT,
        )
    else:
        vectorstore = Qdrant.from_documents(
            documents=documents,
            embedding=embeddings,
            location=config.QDRANT_LOCATION,
            collection_name=config.QDRANT_COLLECTION_NAME,
            distance_func=config.QDRANT_DISTANCE_FUNC
        )

    return vectorstore


# Function to create a conversation chain
def get_conversation_chain(vector_store):
    llm = ChatCohere(model="command-r-plus-08-2024")
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )

    return conversation_chain


# Function to handle user input and display the chat history
def handle_user_input(question):
    response = st.session_state.conversation({'question': question})
    st.session_state.chat_history = response['chat_history']


# Main function to set up the Streamlit app
def main():
    st.set_page_config(page_title='Chat with Your own PDFs', page_icon=':books:', layout="wide")

    st.write(css, unsafe_allow_html=True)
    st.write("""
    <style>
        .stTextInput {
            position: fixed;
            bottom: 30px;
            opacity: 0.8;
        }
    </style>
    """, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False

    st.header('Chat with Your own PDFs :books:')

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.markdown(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.markdown(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

    # Text input at the bottom
    input_container = st.container()
    with input_container:
        question = st.text_input("Ask anything to your PDF:", key="user_input")
        if question and st.session_state.conversation:
            handle_user_input(question)
            st.session_state.user_input = ""  # Clear the input field after submission

    # Sidebar for file upload
    with st.sidebar:
        st.subheader("Upload your Documents Here:")
        pdf_files = st.file_uploader("Choose your PDF Files", type=['pdf'], accept_multiple_files=True)

        ok_button_disabled = not pdf_files or st.session_state.is_processing

        if st.button("OK", disabled=ok_button_disabled):
            st.session_state.is_processing = True
            with st.spinner("Processing your PDFs..."):
                try:
                    raw_text = get_pdf_text(pdf_files)
                    text_chunks = get_chunk_text(raw_text)
                    vector_store = get_vector_store(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vector_store)
                    st.success("Processing complete!")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                finally:
                    st.session_state.is_processing = False


if __name__ == '__main__':
    main()
