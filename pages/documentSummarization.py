import os
import re
import logging
from io import BytesIO
from typing import Any, Dict, List

import pages.utils.config as config  # Import the configuration
import streamlit as st
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import ChatCohere
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain_community.document_loaders import PyPDFLoader
from pages.utils.style import set_page_config
from pypdf import PdfReader


# Set up logging for production-ready error reporting
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

set_page_config()


@st.cache_data
def parse_pdf(file: BytesIO) -> List[str]:
    try:
        pdf = PdfReader(file)
        output = []
        for page in pdf.pages:
            text = page.extract_text()
            # Merge hyphenated words
            text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
            # Fix newlines in the middle of sentences
            text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
            # Remove multiple newlines
            text = re.sub(r"\n\s*\n", "\n\n", text)
            output.append(text)
        return output
    except Exception as e:
        logger.error(f"Error in parse_pdf: {str(e)}")
        raise RuntimeError("Failed to parse PDF file.") from e


@st.cache_data
def text_to_docs(text: str, chunk_size, chunk_overlap) -> List[Document]:
    try:
        logger.info("Converting text to documents.")
        if isinstance(text, str):
            text = [text]
        
        page_docs = [Document(page_content=page) for page in text]

        # Add page numbers as metadata
        for i, doc in enumerate(page_docs):
            doc.metadata["page"] = i + 1

        doc_chunks = []

        for doc in page_docs:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                chunk_overlap=chunk_overlap,
            )
            chunks = text_splitter.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
                )
                doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
                doc_chunks.append(doc)
        return doc_chunks
    except Exception as e:
        logger.error(f"Error in text_to_docs: {str(e)}")
        raise RuntimeError("Failed to convert text to documents.") from e


def custom_summary(docs, llm, custom_prompt, chain_type, num_summaries):
    try:
        custom_prompt = custom_prompt + """:\n {text}"""
        COMBINE_PROMPT = PromptTemplate(template=custom_prompt, input_variables=["text"])
        MAP_PROMPT = PromptTemplate(template="Summarize:\n{text}", input_variables=["text"])

        if chain_type == "map_reduce":
            chain = load_summarize_chain(
                llm, chain_type=chain_type, map_prompt=MAP_PROMPT, combine_prompt=COMBINE_PROMPT
            )
        else:
            chain = load_summarize_chain(llm, chain_type=chain_type)

        summaries = []
        for i in range(num_summaries):
            summary_output = chain({"input_documents": docs}, return_only_outputs=True)["output_text"]
            summaries.append(summary_output)

        return summaries
    except Exception as e:
        logger.error(f"Error in custom_summary: {str(e)}")
        raise RuntimeError("Failed to generate summary.") from e


def main():
    try:
        hide_streamlit_style = """
            <style>
            [data-testid="stToolbar"] {visibility: hidden !important;}
            footer {visibility: hidden !important;}
            </style>
            """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)
        st.title("Document Summarization App")

        llm_name = st.sidebar.selectbox("LLM", ["cohere.command-r-16k", "cohere.command-r-plus", "meta.llama-3-70b-instruct"])
        chain_type = st.sidebar.selectbox("Chain Type", ["map_reduce", "stuff", "refine"])
        chunk_size = st.sidebar.slider("Chunk Size", min_value=20, max_value=5000, step=10, value=2000)
        chunk_overlap = st.sidebar.slider("Chunk Overlap", min_value=5, max_value=5000, step=10, value=200)
        user_prompt = st.text_input("Enter the document summary prompt", value="Compose a brief summary of this text.")
        temperature = st.sidebar.number_input("Set the GenAI Temperature", min_value=0.0, max_value=1.0, step=0.1, value=0.5)
        max_token = st.sidebar.slider("Max Output size", min_value=200, max_value=1000, step=10, value=200)

        uploaded_file = st.file_uploader("**Upload a Pdf file :**", type=["pdf"])

        if uploaded_file:
            if uploaded_file.name.endswith(".txt"):
                doc = parse_txt(uploaded_file)
            elif uploaded_file.name.endswith(".pdf"):
                doc = parse_pdf(uploaded_file)

            pages = text_to_docs(doc, chunk_size, chunk_overlap)

            page_holder = st.empty()
            if pages:
                st.write("PDF loaded successfully")
                with page_holder.expander("File Content", expanded=False):
                    for page in pages:
                        st.write(page.page_content)

            # LLM initialization
            llm = ChatCohere(model="command-r-plus-08-2024", max_token=max_token, temperature=temperature)

            if st.button("Summarize"):
                with st.spinner('Summarizing....'):
                    result = custom_summary(pages, llm, user_prompt, chain_type, 1)
                    st.write("Summary:")
                for summary in result:
                    st.write(summary)

        else:
            st.warning("No file found. Upload a file to summarize!")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        st.error("An error occurred while processing your request. Please try again later.")


if __name__ == "__main__":
    main()
