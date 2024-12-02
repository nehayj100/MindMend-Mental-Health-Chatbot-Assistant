import os
import logging
import streamlit as st
import openai
import ollama
from gtts import gTTS
import chromadb
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE
import time
from langchain.schema import Document

start_time = time.time()
# Initialize Ollama client
ollama_client = openai.Client(base_url="http://127.0.0.1:11434/v1", api_key="EMPTY")

# Directory containing PDF files
DATA_PATH = r"Data/All"
COLLECTION_NAME = "docs"
CHROMA_DB_PATH = "chromadb"

# Helper to initialize ChromaDB client
def init_chroma_client():
    return chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )


# Load documents from PDFs

def load_documents():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Directory does not exist: {DATA_PATH}")
    if not os.listdir(DATA_PATH):
        raise FileNotFoundError(f"No files found in the directory: {DATA_PATH}")

    documents = []
    for filename in os.listdir(DATA_PATH):
        if filename.startswith("."):  # Skip hidden files like .DS_Store
            continue
        filepath = os.path.join(DATA_PATH, filename)
        if filename.endswith(".txt"):  # Process only text files
            with open(filepath, "r", encoding="utf-8") as file:
                # Create a Document object instead of a dictionary
                documents.append(Document(page_content=file.read(), metadata={"source": filename}))
        else:
            raise ValueError(f"Unsupported file format: {filename}")
    
    if not documents:
        raise ValueError("No documents were loaded.")
    return documents


# Split documents into chunks
def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    return text_splitter.split_documents(documents)


# Generate embeddings in batch
def generate_embeddings(text_chunks):
    embeddings = []
    for chunk in text_chunks:
        response = ollama.embeddings(model="llama3.2", prompt=chunk.page_content)
        if "embedding" not in response:
            raise ValueError(f"Embedding generation failed for chunk: {chunk.page_content}")
        embeddings.append((chunk.page_content, response["embedding"]))
    return embeddings


# Perform RAG
def perform_RAG(prompt):
    client = init_chroma_client()

    # Try loading existing collection
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        # Create collection if not exists
        collection = client.create_collection(name=COLLECTION_NAME)

        # Load and process documents
        documents = load_documents()
        chunks = split_text(documents)

        # Batch generate and add embeddings
        embeddings = generate_embeddings(chunks)
        collection.add(
            ids=[str(i) for i in range(len(embeddings))],
            embeddings=[embed[1] for embed in embeddings],
            documents=[embed[0] for embed in embeddings],
        )

    # Generate prompt embedding
    prompt_embedding = ollama.embeddings(model="llama3.2", prompt=prompt)
    if "embedding" not in prompt_embedding:
        raise ValueError("Embedding generation failed for the prompt.")
    
    # Query the database
    results = collection.query(query_embeddings=[prompt_embedding["embedding"]], n_results=5)
    if not results["documents"][0]:
        raise ValueError("No results found.")

    # Combine results and generate response
    combined_data = "\n\n".join(results["documents"][0])
    final_prompt = f'''You are a experienced therapist.Use this data to understand how expert therapists
     handle such situations: {combined_data}. Respond to this prompt: {prompt}. Dont use the examples directly but take inspiration from them to respond'''
    output = ollama.generate(model="llama3.2", prompt=final_prompt)
    return output.get("response", "No response generated.")


# Clear database
def clear_db():
    client = init_chroma_client()
    for collection in client.list_collections():
        client.delete_collection(name=collection.name)
    print("All collections have been cleared.")

def text_to_speech(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    audio_path = "response.mp3"
    tts.save(audio_path)
    return audio_path

# clear_db()

# TODO : ONBOARDING


# TODO : STORING VARIABLES LIKE SOS CONTACT AND DOCTORS EMAIL TO DISK!


all_stm_summary = ""
stm = ""


# Streamlit UI
st.title("MindMend: Your Emergency Therapist")

# Initialize session state for storing chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Input field for user question
question = st.text_input("Enter your question:")

# Handle button press
if st.button("Send"):
    if question.strip():
        with st.spinner("Processing..."):
            start_time = time.time()

            # Get the response (replace with your RAG model call)
            response = perform_RAG(question)
            st.text_area("Answer", value=response, height=200, disabled=True) 

            # Add user question and AI response to the chat history
            st.session_state.messages.append({"role": "user", "content": question})
            st.session_state.messages.append({"role": "ai", "content": response})

            # Convert the AI response to audio and play it
            audio_path = text_to_speech(response)
            audio_file = open(audio_path, "rb")
            st.audio(audio_file.read(), format="audio/mp3")

            # Clean up the generated audio file
            os.remove(audio_path)

            end_time = time.time()
            print("Total time: ", end_time - start_time)

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.write(f"**You:** {msg['content']}")
    else:
        st.write(f"**MindMend:** {msg['content']}")



        # answer = process_input(question + all_stm_summary) # add context prompts properly
        # TODO : For each questions see if its an emergency : SOS calling API

        # TODO : improve prompt!!!
#         stm = f'''
#         question by patient: {question}
#         ansewr by LLM therapist : {answer}
#         '''
#         # TODO # improve prompt
#         stm_summary = invoke_LLM(f"""Summarize this {stm} in a few lines. maximum 4-5. Retain important and notable points. Also capture the sentiment of the question.""")
#         # TODO : check if one more level of summarization is required!!!
#         all_stm_summary += stm_summary
#         st.text_area("Answer", value=answer, height=300, disabled=True) 

# if st.button('End Chat'):
#     # TODO: Ask finally are you Okay, if not: call SOS, add in email that patient wasnt okay even after the chat.
    
#     # create LTM and save
#     # extract old LTM and add new
#     # one more summarization round
#     file_path = "memory/LTM.txt"
#     with open(file_path, "r") as file:
#         content = file.read()

#     # TODO : improve prompt!!!
#     LTM_summary = invoke_LLM(f'''Create a summary in maximum 20 lines of {content} and {all_stm_summary}''')
    
#     content += LTM_summary

#     with open(file_path, "w") as file:
#         file.write(content)

#     # TODO : STM summary to doctor + tell end of the chat

    