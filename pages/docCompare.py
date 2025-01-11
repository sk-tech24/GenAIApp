import os
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pages.utils.lang_utils import ask_to_all_pdfs_sources, create_qa_retrievals
from PIL import Image

# Install pdf2image if not already installed
os.system("python3 -m pip install pdf2image")

# SETUP ------------------------------------------------------------------------
favicon = Image.open("./pages/utils/oracle.webp")
st.set_page_config(
    page_title="PDF Comparison - LLM",
    page_icon=favicon,
    layout="wide",
    initial_sidebar_state="auto",
)

# Sidebar contents ------------------------------------------------------------------------
with st.sidebar:
    st.title("LLM - PDF Comparison App")
    st.markdown("""
    ## About
    This app is a PDF comparison tool (LLM-powered), built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    """)
    st.write("Made with Generative AI")

# ROW 1 ------------------------------------------------------------------------
Title_html = """
    <style>
        .title h1{
          user-select: none;
          font-size: 43px;
          color: white;
          background: repeating-linear-gradient(-45deg, red 0%, yellow 7.14%, rgb(0,255,0) 14.28%, rgb(0,255,255) 21.4%, cyan 28.56%, blue 35.7%, magenta 42.84%, red 50%);
          background-size: 300vw 300vw;
          -webkit-text-fill-color: transparent;
          -webkit-background-clip: text;
          animation: slide 10s linear infinite forwards;
        }
        @keyframes slide {
          0%{ background-position-x: 0%; }
          100%{ background-position-x: 600vw; }
        }
    </style> 
    <div class="title">
        <h1>PDF comparison using Gen AI</h1>
    </div>
"""
components.html(Title_html)

with st.form("basic_form"):
    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf"],
        key="file_upload_widget",
        accept_multiple_files=True,
    )

    # Questions
    questions = [st.text_input(f"Question {i+1}", key=f"{i+1}_question") for i in range(4)]

    submit_btn = st.form_submit_button("Start Processing")

    if submit_btn:
        # Validate inputs
        if not any(questions):
            st.warning("Give at least one question")
            st.stop()

        if not uploaded_files:
            st.warning("Upload at least one PDF file")
            st.stop()

        # Process files and questions
        with st.spinner("Creating embeddings..."):
            try:
                st.session_state.qa_retrievals = create_qa_retrievals(uploaded_files)
                st.session_state.questions = [q for q in questions if q]  # Filter empty questions
            except Exception as e:
                st.error("An error occurred while creating embeddings.")
                st.exception(e)
                st.stop()

        st.success("Embeddings created!", icon="✅")

        with st.spinner("Performing analysis..."):
            try:
                results = []
                for question in st.session_state.questions:
                    if question:
                        results.extend(ask_to_all_pdfs_sources(question, st.session_state.qa_retrievals))

                st.session_state.data = results
            except Exception as e:
                st.error("An error occurred during analysis.")
                st.exception(e)
                st.stop()

        st.success("Analysis complete!", icon="✅")

        with st.spinner("Formatting results..."):
            try:
                df = pd.DataFrame(st.session_state.data)
                if df.duplicated(subset=["query", "source_document"]).any():
                    st.warning("Duplicate entries found. Removing duplicates...")
                    df = df.drop_duplicates(subset=["query", "source_document"])

                st.table(df.pivot(index="query", columns="source_document", values="response"))
            except Exception as e:
                st.error("An error occurred while formatting the results.")
                st.exception(e)
                st.stop()

        st.balloons()
