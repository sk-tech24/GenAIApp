# pages/utils/utils.py
import os

import streamlit as st
from PIL import Image


def set_page_config():
    os.environ.setdefault('COHERE_API_TOKEN', 'JmNhbEWy3qQIYLeTWwVqZPPVH3xzteNzgBDUqm8y')
    favicon = Image.open("./pages/utils/oracle.webp")
    st.set_page_config(
        page_title="Gen AI Application",
        page_icon=favicon,
        layout="wide",
        initial_sidebar_state="auto",
    )

    hide_streamlit_style = """
        <style>
        [data-testid="stToolbar"] {visibility: hidden !important;}
        footer {visibility: hidden !important;}
        </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
