from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-7B-Chat-GGUF")
model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7B-Chat-GGUF")

llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

def generate_response(vector_store, query):
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vector_store.as_retriever())
    return qa.run(query)