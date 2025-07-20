import os
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import tempfile

# LLM model to use with Ollama
LLM = "deepseek-coder:6.7b-instruct-q4_K_M"

# Prompt template for answering questions
template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

# Temp directory for PDFs
pdfs_directory = "chat-with-pdf/pdfs/"
os.makedirs(pdfs_directory, exist_ok=True)

# Load HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize Ollama LLM
model = OllamaLLM(model=LLM)

# Vector store (global so it's reusable after upload)
vector_store = None


def upload_pdf(file):
    """Save the uploaded PDF to a temporary file."""
    try:
        file_path = os.path.join(pdfs_directory, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None


def load_pdf(file_path):
    """Load and return documents from the PDF."""
    try:
        loader = PDFPlumberLoader(file_path)
        return loader.load()
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return None


def split_text(documents):
    """Split large PDF text into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    return text_splitter.split_documents(documents)


def index_docs(documents):
    """Index documents in a Chroma vector store."""
    global vector_store
    vector_store = Chroma.from_documents(documents, embeddings)


def retrieve_docs(query):
    """Retrieve similar documents based on a question."""
    return vector_store.similarity_search(query)


def answer_question(question, documents):
    """Generate an answer using context and the LLM."""
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})


# ---------------------- Streamlit UI ---------------------- #
st.set_page_config(page_title="Chat with Your PDF")
st.title("📄 Chat with Your PDF using LLM + Local Embeddings")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    file_path = upload_pdf(uploaded_file)

    if file_path:
        st.success(f"File uploaded: {uploaded_file.name}")

        with st.spinner("Processing PDF..."):
            documents = load_pdf(file_path)
            if documents:
                chunks = split_text(documents)
                index_docs(chunks)
                st.success("PDF processed and indexed! Ask questions below.")

        question = st.chat_input("Ask a question about the PDF:")
        if question:
            st.chat_message("user").write(question)
            with st.spinner("Searching for answer..."):
                results = retrieve_docs(question)
                if results:
                    answer = answer_question(question, results)
                    st.chat_message("assistant").write(answer)
                else:
                    st.chat_message("assistant").write("No relevant information found.")
