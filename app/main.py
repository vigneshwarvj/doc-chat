import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from streamlit_chat import message
from backend.embedder import embed_and_store, embed_query
from backend.document_loader import load_documents
from backend.llama2_generator import generate_response

load_dotenv()

st.set_page_config(page_title="LLaMA 2 RAG Chatbot", page_icon="🦙", layout="wide")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🦙 LLaMA 2 RAG Chatbot")

with st.sidebar:
    st.header("📄 Upload Documents")
    uploaded_files = st.file_uploader("Upload PDF, DOCX, or TXT files", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    if st.button("Process Documents"):
        if uploaded_files:
            texts = load_documents(uploaded_files)
            st.session_state.vector_store = embed_and_store(texts)
            st.success("✅ Documents processed and indexed.")
        else:
            st.warning("⚠️ Please upload at least one document.")

if st.session_state.vector_store:
    user_input = st.text_input("Ask a question about the documents:")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        result = generate_response(st.session_state.vector_store, user_input)
        st.session_state.chat_history.append({"role": "assistant", "content": result})
        for i, chat in enumerate(st.session_state.chat_history[::-1]):
            message(chat["content"], is_user=chat["role"] == "user", key=f"msg_{i}")
else:
    st.info("👈 Upload and process documents to begin.")