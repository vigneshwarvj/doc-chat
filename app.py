import streamlit as st
from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load Llama 2 7B (local model path or HuggingFace model repo)
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"  # Hugging Face repo

@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype="auto")
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.2,
    )
    return HuggingFacePipeline(pipeline=pipe)

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def main():
    st.title("Chat with your PDF (Llama 2 + FAISS + SentenceTransformers)")

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file is not None:
        # Load PDF
        loader = PDFPlumberLoader(uploaded_file)
        documents = loader.load()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)

        # Load embedding model and embed docs
        embed_model = load_embedding_model()
        vector_store = FAISS.from_documents(docs, embed_model)

        st.success("PDF loaded and indexed!")

        # Load Llama 2 model
        llm = load_model()

        question = st.text_input("Ask a question about the PDF:")

        if question:
            # Retrieve relevant docs
            related_docs = vector_store.similarity_search(question, k=3)
            context = "\n\n".join([doc.page_content for doc in related_docs])

            prompt = f"""
You are a helpful assistant. Use the context below to answer the question. If you don't know, say so.

Context:
{context}

Question: {question}

Answer:"""

            # Generate answer from Llama 2
            response = llm(prompt)
            st.markdown("### Answer:")
            st.write(response)

if __name__ == "__main__":
    main()
