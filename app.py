import os
from pathlib import Path

import openai
import streamlit as st

st.set_page_config(page_title="Ask Nour Cafe", page_icon="☕")

# On Streamlit Cloud, secrets live in st.secrets. Expose them as env vars
# so src/config.py can read them. Locally, .env is used instead.
try:
    for key, value in st.secrets.items():
        if isinstance(value, str):
            os.environ.setdefault(key, value)
except Exception:
    pass

from src.config import INDEX_PATH, TOP_K
from src.build_index import main as build_index
from src.vector_store import load_index
from src.retrieval import retrieve
from src.generate import generate_answer


@st.cache_resource(show_spinner="Preparing the knowledge base...")
def get_index():
    if not Path(INDEX_PATH).exists():
        build_index()
    return load_index()


def check_password() -> bool:
    expected = os.getenv("APP_PASSWORD")
    if not expected or st.session_state.get("authed"):
        return True
    pwd = st.text_input("Password", type="password")
    if pwd:
        if pwd == expected:
            st.session_state["authed"] = True
            st.rerun()
        st.error("Wrong password.")
    return False


st.title("Ask Nour Cafe ☕")
st.caption("Answers come only from the café's internal policy document.")

if not check_password():
    st.stop()

chunks, vectors = get_index()

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources used"):
                for text in m["sources"]:
                    st.text(text)

question = st.chat_input("Ask a question about the document", max_chars=500)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                retrieved = retrieve(question, chunks, vectors, top_k=TOP_K)
                answer = generate_answer(question, retrieved)
            sources = [text for _, _, text in retrieved]
            st.markdown(answer)
            with st.expander("Sources used"):
                for text in sources:
                    st.text(text)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except openai.OpenAIError:
            st.error("Something went wrong talking to the AI service. Please try again.")