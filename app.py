import streamlit as st
import tempfile
from utils import (
    get_pdf_pages,
    get_chroma_vectors_db,
    initialize_llm,
    get_question_answer_chain,
    process_llm_response,
)

st.set_page_config(page_title="PDF Chatbot with Llama2", layout="wide")
st.title("📄🦙 Chat with your PDF using Llama2 + LangChain")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Processing your PDF..."):
        # Save the uploaded PDF to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # Load and split the PDF pages
        pages = get_pdf_pages(tmp_path)

        # Generate embeddings and build vector store
        vector_db = get_chroma_vectors_db(pages)

        # Load the Llama2 model via Ollama
        llm = initialize_llm()

        # Build the question-answer chain
        qa_chain = get_question_answer_chain(vector_db, llm)

        st.success("PDF processed! You can now ask questions about it.")

        # User input for questions
        question = st.text_input("Ask a question about the PDF:")
        if question:
            with st.spinner("Generating answer..."):
                response = qa_chain({"query": question})
                answer = process_llm_response(response)
                st.markdown("### 🧠 Answer:")
                st.write(answer)
