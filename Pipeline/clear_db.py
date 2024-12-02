import os
import openai
import ollama
import chromadb
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE
import time

start_time = time.time()
# Initialize Ollama client
ollama_client = openai.Client(base_url="http://127.0.0.1:11434/v1", api_key="EMPTY")

# Directory containing PDF files
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


# Clear database
def clear_db():
    client = init_chroma_client()
    for collection in client.list_collections():
        client.delete_collection(name=collection.name)
    print("All collections have been cleared.")

# Example Usage
clear_db()
# response = perform_RAG("SHould alice be scolded?")
# print(response)
end_time = time.time()

print("Total time: ", end_time-start_time)