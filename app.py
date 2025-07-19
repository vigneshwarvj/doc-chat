import os
import streamlit as st
from streamlit_ace import st_ace
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from streamlit_chat import message
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS, SupabaseVectorStore
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
# from supabase.client import Client, create_client


# Instantiate the model. Callbacks support token-wise streaming
# model = GPT4All(model="./models/ggml-gpt4all-j-v1.3-groovy.bin", n_ctx=512, n_threads=8)

# Generate text
# response = model("Once upon a time, ")


def get_pdf_text(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunk(raw_texts):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=2000, chunk_overlap=500, length_function=len
    )
    chunk = text_splitter.split_text(raw_texts)
    return chunk


def get_vector_store(text_chunks):
    # supabase_url = os.environ.get("SUPABASE_URL")
    # supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
    # supabase: Client = create_client(supabase_url, supabase_key)

    embeddings = OpenAIEmbeddings()
    # print(text_chunks)
    # vector_store = SupabaseVectorStore.from_texts(
    #     text_chunks, embeddings, client=supabase
    # )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store


def get_conversation_chain(vector_store):
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vector_store.as_retriever(), memory=memory
    )
    return conversation_chain


def handle_user_question(user_question):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    # Update the chat history with the user's new question
    st.session_state.chat_history.append({"role": "user", "content":  user_question})
    
    # response = st.session_state.conversation({
    #     "question": user_question
    # })

    response = st.session_state.conversation.invoke({
    "question": user_question
    })

    
    # incase to log the responses
    # st.write(response)
    st.session_state.chat_history.append({"role": "assistant", "content": response['answer']})

    reversed_chat_history = st.session_state.chat_history[::-1]

    for i, chat in enumerate(reversed_chat_history): 
        is_user = chat['role'] == 'user'
        
        if chat['role'] == "assistant":
            # Bot response
            message(chat['content'], is_user=is_user, key=f"chat_message_{i}")
        else:
            # User questions
            message(chat['content'], is_user=is_user, key=f"chat_message_{i}")


def main():
    load_dotenv()

    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header("Chat with PDF :books:")
    user_question = st.text_input("Ask any question based on the content from the document you just uploaded")
    # user_question = st_ace(language="markdown",show_gutter=False, theme="twilight", keybinding="vscode")

    if user_question:
        handle_user_question(user_question)

    # message("Hello There, How can I assist you")

    with st.sidebar:
        st.header("Your document")
        pdf_docs = st.file_uploader("Upload you pdf file", accept_multiple_files=True, type=["pdf"])

        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                # get text chunks
                text_chunk = get_text_chunk(raw_texts=raw_text)

                # create vector store
                vector_store = get_vector_store(text_chunk)

                # create a conversation chain
                st.session_state.conversation = get_conversation_chain(vector_store)


if __name__ == "__main__":
    main()
