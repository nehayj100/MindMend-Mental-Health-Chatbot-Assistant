import ollama
import chromadb

from langchain_community.document_loaders import PyPDFDirectoryLoader  # Importing PDF loader from Langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Importing text splitter from Langchain
from langchain.schema import Document  # Importing Document schema from Langchain
from chromadb.config import Settings

# Directory to your pdf files:
DATA_PATH = r"papers"

def load_documents():
    """
    Load PDF documents from the specified directory using PyPDFDirectoryLoader.

    Returns:
        List of Document objects: Loaded PDF documents represented as Langchain Document objects.
    """
    document_loader = PyPDFDirectoryLoader(DATA_PATH)  # Initialize PDF loader with specified directory
    return document_loader.load()  # Load PDF documents and return them as a list of Document objects

# documents = load_documents()


def split_text(documents: list[Document]):
    """
    Split the text content of the given list of Document objects into smaller chunks.

    Args:
        documents (list[Document]): List of Document objects containing text content to split.

    Returns:
        list[Document]: List of Document objects representing the split text chunks.
    """
    # Initialize text splitter with specified parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,  # Size of each chunk in characters
        chunk_overlap=100,  # Overlap between consecutive chunks
        length_function=len,  # Function to compute the length of the text
        add_start_index=True,  # Flag to add start index to each chunk
    )
    # Split documents into smaller chunks using text splitter
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # Print example of page content and metadata for a chunk
    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks  # Return the list of split text chunks

# documents = [
#   "Llamas are members of the camelid family meaning they're pretty closely related to vicuñas and camels",
#   "Llamas were first domesticated and used as pack animals 4,000 to 5,000 years ago in the Peruvian highlands",
#   "Llamas can grow as much as 6 feet tall though the average llama between 5 feet 6 inches and 5 feet 9 inches tall",
#   "Llamas weigh between 280 and 450 pounds and can carry 25 to 30 percent of their body weight",
#   "Llamas are vegetarians and have very efficient digestive systems",
#   "Llamas live to be about 20 years old, though some only live for 15 years and others live to be 30 years old",
# ]

from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings

client = chromadb.PersistentClient(
    path="chromadb",
    settings=Settings(),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE,
)

# client = chromadb.Client(settings={"chromadb.storage": "sqlite", "chromadb.storage.path": "chromadb.db"})
# Collection name
collection_name = "docs"

# Try to load the collection; if it doesn't exist, create it
try:
    # Attempt to get the collection by name
    collection = client.get_collection(name=collection_name)
    print(f"Collection '{collection_name}' loaded successfully.")
except Exception as e:
    # If collection doesn't exist, create it
    print(f"Collection '{collection_name}' not found. Creating a new one.")
    collection = client.create_collection(name=collection_name)

    documents = load_documents()  # Load documents from a source
    chunks = split_text(documents)  # Split documents into manageable chunks

    # store each document in a vector embedding database
    for i, chunk in enumerate(chunks):
      d = chunk.page_content
      print(f"Chunk {i}: {d}")
      response = ollama.embeddings(model="llama3.1", prompt=d)
      embedding = response["embedding"]
      # print(f"embedding: {embedding}")
      collection.add(
        ids=[str(i)],
        embeddings=[embedding],
        documents=[d]
      )

# an example prompt
prompt = "How does chain of thought prompting work?"

# prompt = "What's the score of llama 3 8B on MATH (0-shot, CoT)?"

# generate an embedding for the prompt and retrieve the most relevant doc
embedding = ollama.embeddings(
  prompt=prompt,
  model="llama3.1"
)["embedding"]
# print(f"embedding: length {len(embedding)})\n{embedding}")

results = collection.query(
  query_embeddings=[embedding],
  n_results=5
)

# print(f"results:\n{results}")

# data = results['documents'][0][0]
# Extract the documents from the results
data1 = results['documents'][0][0]
data2 = results['documents'][0][1]
data3 = results['documents'][0][2]
data4 = results['documents'][0][3]
data5 = results['documents'][0][4]

# Combine the data into a single string
combined_data = f"{data1}\n\n{data2}\n\n{data3}\n\n{data4}\n\n{data5}"

prompt = f"Using this data: {combined_data}. Respond to this prompt: {prompt}"

print(f"prompt: {prompt}")

# generate a response combining the prompt and data we retrieved in step 2
output = ollama.generate(
  model="llama3.1",
  prompt=prompt
)

print(f"response:\n{output['response']}")
