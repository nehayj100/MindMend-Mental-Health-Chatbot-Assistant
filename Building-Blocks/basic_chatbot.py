from langchain_ollama import OllamaLLM
import streamlit as st

llm = OllamaLLM(model="llama3.2:latest")
st.title("Chatbot using Llama3")

prompt = st.text_area("Enter your prompt:")

if st.button("Generate"):
    if prompt:
        with st.spinner("Generating response..."):
            st.write_stream(llm.stream(prompt, stop=['<|eot_id|>']))