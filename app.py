import streamlit as st
import PyMuPDF  # for PDF processing
import docx2txt
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from pathlib import Path
import torch

# Initialize models
@st.cache_resource
def load_models():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    model_name = "meta-llama/Llama-2-7b-chat-hf"  # Replace with actual model path
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = AutoModelForCausalLM.from_pretrained(model_name)
    return embedder, tokenizer, llm

# Document processing functions
def extract_text_from_pdf(file_path):
    doc = PyMuPDF.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def process_document(file_path):
    ext = Path(file_path).suffix.lower()
    if ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext == '.docx':
        return extract_text_from_docx(file_path)
    elif ext == '.txt':
        return extract_text_from_txt(file_path)
    else:
        raise ValueError("Unsupported file format")

# Chunk text for better processing
def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        current_length += len(word) + 1
        current_chunk.append(word)
        
        if current_length > chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
            
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

# Create and store embeddings
class VectorStore:
    def __init__(self):
        self.index = faiss.IndexFlatL2(384)  # Dimension for all-MiniLM-L6-v2
        self.text_chunks = []
        
    def add_document(self, text, embedder):
        chunks = chunk_text(text)
        embeddings = embedder.encode(chunks)
        
        self.text_chunks.extend(chunks)
        embeddings = np.array(embeddings).astype('float32')
        self.index.add(embeddings)
        
    def search(self, query, embedder, k=3):
        query_embedding = embedder.encode([query])[0].astype('float32')
        distances, indices = self.index.search(np.array([query_embedding]), k)
        return [(self.text_chunks[i], distances[0][j]) for j, i in enumerate(indices[0])]

# Generate response using Llama 2
def generate_response(query, context, tokenizer, llm):
    prompt = f"""Context: {context}
    
Question: {query}

Answer: """
    
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = llm.generate(**inputs, max_length=500, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit frontend
def main():
    st.title("Document Q&A System")
    
    # Initialize models and vector store
    embedder, tokenizer, llm = load_models()
    vector_store = VectorStore()
    
    # File upload
    uploaded_files = st.file_uploader("Upload documents", 
                                    type=['pdf', 'docx', 'txt'], 
                                    accept_multiple_files=True)
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Save uploaded file temporarily
            with open(uploaded_file.name, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            # Process document and add to vector store
            text = process_document(uploaded_file.name)
            vector_store.add_document(text, embedder)
            
            # Clean up
            os.remove(uploaded_file.name)
        
        st.success("Documents processed successfully!")
    
    # Query input
    query = st.text_input("Ask a question about your documents:")
    
    if query and st.button("Get Answer"):
        # Search for relevant context
        results = vector_store.search(query, embedder)
        context = "\n".join([result[0] for result in results])
        
        # Generate response
        with st.spinner("Generating answer..."):
            response = generate_response(query, context, tokenizer, llm)
        
        st.write("**Answer:**")
        st.write(response)
        
        st.write("**Relevant Context:**")
        for chunk, score in results:
            st.write(f"Score: {score:.4f}")
            st.write(chunk)
            st.write("---")

if __name__ == "__main__":
    main()
