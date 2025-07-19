from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import streamlit as st

# Load PDF and split text
loader = PDFPlumberLoader("example.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# Embed text chunks using SentenceTransformers
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(docs, embedding_model)

# Load Llama 2 locally via Hugging Face (adjust model path or use HF API)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.2,
)

llm = HuggingFacePipeline(pipeline=pipe)

# Streamlit app
st.title("Chat with your PDF")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file:
    docs = loader.load()
    chunks = text_splitter.split_documents(docs)
    vector_store = FAISS.from_documents(chunks, embedding_model)

    question = st.text_input("Ask a question about the PDF:")
    if question:
        results = vector_store.similarity_search(question, k=3)
        context = "\n".join([doc.page_content for doc in results])
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"

        response = llm(prompt)
        st.write(response)
