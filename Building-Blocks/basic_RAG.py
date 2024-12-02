import os
import openai
import ollama
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
DATA_PATH = r"test-data"
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
    final_prompt = f"Using this data: {combined_data}. Respond to this prompt: {prompt}"
    output = ollama.generate(model="qwen2:latest", prompt=final_prompt)
    return output.get("response", "No response generated.")


# Clear database
def clear_db():
    client = init_chroma_client()
    for collection in client.list_collections():
        client.delete_collection(name=collection.name)
    print("All collections have been cleared.")


# Example Usage
clear_db()
response = perform_RAG("SHould alice be scolded?")
print(response)
end_time = time.time()

print("Total time: ", end_time-start_time)