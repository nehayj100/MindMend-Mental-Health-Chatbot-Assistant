import os
import openai
import ollama
import chromadb
from langchain_community.document_loaders import PyPDFDirectoryLoader  # Importing PDF loader from Langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Importing text splitter from Langchain
from langchain.schema import Document  # Importing Document schema from Langchain
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE


# Initialize Ollama client
ollama_client = openai.Client(base_url="http://127.0.0.1:11434/v1", api_key="EMPTY")

# Directory containing PDF files
DATA_PATH = r"test-data"

def load_documents():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Directory does not exist: {DATA_PATH}")
    if not os.listdir(DATA_PATH):
        raise FileNotFoundError(f"No files found in the directory: {DATA_PATH}")
    
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = document_loader.load()
    if not documents:
        raise ValueError("Document loading failed. No documents were loaded.")
    
    print(f"Loaded {len(documents)} documents.")
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,  # Chunk size in characters
        chunk_overlap=100,  # Overlap between consecutive chunks
        length_function=len,  # Function to compute text length
        add_start_index=True,  # Add start index metadata
    )

    chunks = text_splitter.split_documents(documents)

    if len(chunks) <= 10:
        raise ValueError(f"Not enough chunks. Found only {len(chunks)} chunks.")
    return chunks


def perform_RAG(prompt):
    # Initialize ChromaDB Persistent Client
    rag_client = chromadb.PersistentClient(
        path="chromadb",
        settings=Settings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )

    # Define collection name
    collection_name = "docs"

    try:
        collection = rag_client.get_collection(name=collection_name)
        print(f"Collection '{collection_name}' loaded successfully.")
    except Exception:
        print(f"Collection '{collection_name}' not found. Creating a new one.")
        collection = rag_client.create_collection(name=collection_name)

        # Load and process documents
        documents = load_documents()
        chunks = split_text(documents)

        # Store each chunk in the vector database
        for i, chunk in enumerate(chunks):
            response = ollama.embeddings(model="llama3.2", prompt=chunk.page_content)
            if "embedding" not in response:
                raise ValueError(f"Embedding generation failed for chunk: {chunk.page_content}")
            collection.add(
                ids=[str(i)],
                embeddings=[response["embedding"]],
                documents=[chunk.page_content],
            )

    # Generate embedding for the user prompt
    embedding_response = ollama.embeddings(prompt=prompt, model="llama3.2")
    if "embedding" not in embedding_response:
        raise ValueError(f"Embedding generation failed for the prompt: {prompt}")
    embedding = embedding_response["embedding"]

    # Query the collection
    results = collection.query(query_embeddings=[embedding], n_results=5)
    if len(results["documents"][0]) < 5:
        raise ValueError("Insufficient results from the query.")

    # Combine retrieved data
    combined_data = "\n\n".join(results["documents"][0][:5])
    final_prompt = f"Using this data: {combined_data}. Respond to this prompt: {prompt}"

    # print(f"Final prompt sent to Ollama:\n{final_prompt}")

    # Generate the final response
    output = ollama.generate(model="llama3.2", prompt=final_prompt)
    if "response" not in output:
        raise ValueError("Failed to generate response from Ollama.")

    # print(f"Response:\n{output['response']}")
    return output["response"]


def clear_db():
    # Initialize ChromaDB Persistent Client
    client = chromadb.PersistentClient(
        path="chromadb",
        settings=Settings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )

    # Delete all collections
    collections = client.list_collections()
    for collection in collections:
        client.delete_collection(name=collection.name)
    print("All collections have been cleared.")


# Clear database and run Retrieval-Augmented Generation
# clear_db()
response = perform_RAG("where is alice?")
print(response)
