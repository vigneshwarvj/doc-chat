import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import LlamaCpp

# Change these to your local Llama 2 model path and SentenceTransformer model name
LLAMA_MODEL_PATH = "/path/to/llama-2-7b-chat.ggmlv3.q8_0.bin"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temp file and return the path"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        return tmp_file.name

def main():
    st.title("Chat with Your PDF (Llama 2 + FAISS + SentenceTransformers)")

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file is not None:
        # Save to temp file
        file_path = save_uploaded_file(uploaded_file)

        # Load PDF content
        loader = PDFPlumberLoader(file_path)
        documents = loader.load()

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs_chunks = text_splitter.split_documents(documents)

        # Initialize embeddings and vector store (FAISS)
        embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vector_store = FAISS.from_documents(docs_chunks, embeddings)

        # Initialize LlamaCpp LLM
        llm = LlamaCpp(model_path=LLAMA_MODEL_PATH, n_ctx=2048, temperature=0.7)

        # Build retrieval-based QA chain
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())

        # Ask questions
        question = st.text_input("Ask a question about the uploaded PDF:")

        if question:
            with st.spinner("Generating answer..."):
                answer = qa_chain.run(question)
            st.write("**Answer:**")
            st.write(answer)

if __name__ == "__main__":
    main()
