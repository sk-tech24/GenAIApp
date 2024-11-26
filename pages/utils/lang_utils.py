import oracledb
import pages.utils.config as config  # Import the configuration
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_cohere import ChatCohere
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_community.vectorstores.oraclevs import OracleVS
from langchain_community.vectorstores.utils import DistanceStrategy
from PyPDF2 import PdfReader


# Function to extract text from PDFs
def extract_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = "".join(page.extract_text() for page in reader.pages)
    return text

# Function to process text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

# Function to process PDF files and create QA retrieval chains
def create_qa_retrievals(pdf_file_list: list):
    qa_retrievals = []
    embeddings = CohereEmbeddings(
        model=config.EMBEDDING_MODEL,  # Replace with your Cohere model name
        user_agent="langchain"
    )

    for pdf in pdf_file_list:
        try:
            raw_text = extract_pdf_text(pdf)
            if not raw_text:
                st.warning(f"No text found in {pdf.name}. Skipping...")
                continue

            text_chunks = get_text_chunks(raw_text)
            documents = [
                Document(page_content=chunk, metadata={"source": pdf.name, "page": idx + 1})
                for idx, chunk in enumerate(text_chunks)
            ]

            # Select Vector Database
            if config.DB_TYPE == "oracle":
                connection = oracledb.connect(
                    user=config.ORACLE_USERNAME,
                    password=config.ORACLE_PASSWORD,
                    dsn=config.ORACLE_DSN
                )
                vector_db = OracleVS.from_documents(
                    documents=documents,
                    embedding=embeddings,
                    client=connection,
                    table_name=config.ORACLE_TABLE_NAME,
                    distance_strategy=DistanceStrategy.DOT_PRODUCT
                )
                st.success(f"Successfully saved {pdf.name} to Oracle vector DB.")
            else:
                vector_db = Qdrant.from_documents(
                    documents=documents,
                    embedding=embeddings,
                    location=config.QDRANT_LOCATION,
                    collection_name=pdf.name,
                )
                st.success(f"Successfully saved {pdf.name} to vector DB.")

            # Define QA Chain
            llm = ChatCohere(model="command-r-plus-08-2024")
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 2}),
                return_source_documents=True,
            )

            qa_retrievals.append(qa_chain)

        except Exception as e:
            st.error(f"Error processing {pdf.name}: {str(e)}")
            continue

    return qa_retrievals

# Function to query all QA chains
def ask_to_all_pdfs_sources(query: str, qa_retrievals):
    responses = []
    progress_text = f"Asking '{query}' to all PDFs"
    my_bar = st.progress(0, text=progress_text)

    for idx, qa in enumerate(qa_retrievals):
        result = qa({"query": query})
        response = {
            "query": query,
            "response": result.get("result", "No response"),
            "source_document": result["source_documents"][0].metadata["source"] if result.get("source_documents") else "Unknown"
        }
        responses.append(response)

        percent_complete = (idx + 1) * 100 / len(qa_retrievals)
        my_bar.progress(int(percent_complete), text=progress_text)

    return responses
