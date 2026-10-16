import streamlit as st

st.write("1. streamlit ok")

import openai
st.write("2. openai ok")

from src.config import INDEX_PATH, TOP_K
st.write("3. config ok")

from src.build_index import main as build_index
from src.vector_store import load_index
from src.retrieval import retrieve
from src.generate import generate_answer
st.write("4. all imports ok")

from pathlib import Path
st.write("5. index exists:", Path(INDEX_PATH).exists())
