import os
import re, json, time
from pydantic import BaseModel, Field
import openai
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
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import smtplib
import json
from twilio.rest import Client

start_time = time.time()

# Initialize Ollama client
ollama_client = openai.Client(base_url="http://127.0.0.1:11434/v1", api_key="EMPTY")

# Directory containing PDF files
DATA_PATH = r"Data/All"
COLLECTION_NAME = "docs"
CHROMA_DB_PATH = "chromadb"
stop = ['Observation:', 'Observation ']

doctor_email_path = "onboarding-details/doctors-email.txt"
user_fullname_path = "onboarding-details/user-full-name.txt"
doctor_name_path = "onboarding-details/doctor-name.txt"
sos_contact_name_path = "onboarding-details/sos-contact-name.txt"
sos_contact_number_path = "onboarding-details/sos-contact-number.txt"
user_contact_number_path = "onboarding-details/user-contact-number.txt"

password_path = "confidential/email_pass.txt"
# Open and read the file for email password
with open(password_path, 'r') as file:
    passkey = file.read()  # Read the entire content of the file 

LTM_file_path = "memory/LTM.txt"
# STM_file_path = "memory/STM.txt"

with open(LTM_file_path, "r") as file:
    current_LTM = file.read()

#calling API credentials

call_account_sid_path = "confidential/twillio_sid.txt"
call_auth_token_path = "confidential/twillio_auth.txt"
twilio_number_path = "confidential/twillio_num.txt"
to_number_path = "confidential/sos_contact.txt"


with open(call_account_sid_path, "r") as file:
    account_sid = file.read()

with open(call_auth_token_path, "r") as file:
    auth_token = file.read()

with open(twilio_number_path, "r") as file:
    twilio_number = file.read()

# with open(to_number_path, "r") as file:
#     to_number = file.read()


# Helper to initialize ChromaDB client
def init_chroma_client():
    return chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )


# functions for emails

def invoke_llm(prompt:str) -> str:
    try:
        response = ollama_client.completions.create(
            model="llama3.2:latest",
            prompt=prompt,
            stop=stop,
        )
        output = response.choices[0].text
    except Exception as e:
        output = f"Exception: {e}"

    return output

def find_email(query: str) -> str:
    file_path = "onboarding-details/doctors-email"
    try:
        with open(file_path, 'r') as file:
            file_content = file.read().strip()  
        return file_content
    except FileNotFoundError:
        "The file 'doctors-email' was not found in the specified folder."
    except Exception as e:
        f"An error occurred: {str(e)}" 

def send_email_internal(to_addr: str, subject: str, body: str) -> str:
    
    # SMTP server configuration
    smtp_server = "smtp.gmail.com"  # This might need to be updated
    smtp_port = 587  # or 465 for SSL or 587
    username = "testtestertamu@gmail.com"
    password = f"{passkey}"
    from_addr = "testtestertamu@gmail.com"

    cc_addr = "xxx"

    # Email content

    # Setting up the MIME
    message = MIMEMultipart()
    message["From"] = from_addr
    message["To"] = to_addr
    message["Subject"] = subject
    # message["Cc"] = cc_addr  # Add CC here
    message.attach(MIMEText(body, "plain"))

    recipients = [to_addr, cc_addr]  # List of all recipients

    # Send the email
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()  # Secure the connection
            server.login(username, password)
            text = message.as_string()
            server.sendmail(from_addr, recipients, text)
            output = "Email successfully sent!"
    except Exception as e:
        output = f"Failed to send email: {e}"
    print(output)
    return output

def get_doctors_email():
    doctor_email_file_path = "onboarding-details/doctors-email.txt"
    try:
        with open(doctor_email_file_path, 'r') as file:
            file_content = file.read()  # Read and strip any extra whitespace
        return file_content
    except FileNotFoundError:
        "The file 'doctors-email' was not found in the specified folder."
    except Exception as e:
        f"An error occurred: {str(e)}"

def send_email(body_content):
    to_addr = get_doctors_email()
    user_full_name = get_user_full_name()
    now = datetime.now()
    formatted_date_time = now.strftime("%Y-%m-%d %H:%M:%S")
    print("user name is: ", get_user_full_name())
    print("Doctors email is: ", get_doctors_email())
    subject = f'''Summary of conversation with {user_full_name} on {formatted_date_time}'''
    body = invoke_llm(f'''summarize this text: {body_content} as an email body. This email is a summary of 
                        conversation between the user and a mental health chatbot called MinMend''')

    output = send_email_internal(to_addr, subject, body)
    # print(output)


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

def emergency_calling():
    # Initialize the Twilio client
    client = Client(account_sid, auth_token)
    user_full_name = get_user_full_name()
    # Make the call
    call = client.calls.create(
        to = get_sos_contact_number(),
        from_= twilio_number,
        twiml=f'''<Response><Say>Hi, I am MindMend, {user_full_name} who has chosen you as your SOS contact is in a posisble emergency and needs your help! I have emailed you her conversation with me. Pleaes contact her urgently!!</Say></Response>'''
    )

    # Print call details
    print(f"Call SID: {call.sid}")
    print(f"Call Status: {call.status}")

def save_onboarding_info(user_full_name,doctor_name, doctor_email, sos_contact_name, sos_contact_number, user_contact_number):
    with open(user_fullname_path, 'w') as file:
        file.write(user_full_name)
    with open(doctor_name_path, 'w') as file:
        file.write(doctor_name)
    with open(doctor_email_path, 'w') as file:
        file.write(doctor_email)
    with open(sos_contact_name_path, 'w') as file:
        file.write(sos_contact_name)
    with open(sos_contact_number_path, 'w') as file:
        file.write(sos_contact_number)
    with open(user_contact_number_path, 'w') as file:
        file.write(user_contact_number)

def get_user_full_name():
    with open(user_fullname_path, 'r') as file:
        return file.read()
def get_doctor_name():
    with open(doctor_name_path, 'r') as file:
        return file.read()
def get_sos_contact_name():
    with open(sos_contact_name_path, 'r') as file:
        return file.read()
def get_sos_contact_number():
    with open(sos_contact_number_path, 'r') as file:
        return file.read()
def get_user_contact_number():
    with open(user_contact_number_path, 'r') as file:
        return file.read()

# clear_db()


# TODO : improve prompt!!!
all_stm_summary = ""
#invoke_llm(f'''Summarize this text in 5-6 lines. make sure to include all important points: {current_LTM}''')
stm = ""

# Set the page config as the first Streamlit command
st.set_page_config(page_title="MindMend : Let's talk!", layout="wide")

# Onboarding and Chat Tabs using Streamlit tabs
tabs = st.tabs(["Chat", "Onboarding"])

# Onboarding Tab
with tabs[1]:
    if "page" not in st.session_state:
        st.session_state.page = "form"  # Default to the form page

    # Form page for user input
    if st.session_state.page == "form":
        st.title("Welcome to MindMend!")
        st.header("Onboarding Process")
        st.write("This is where we will guide you through the onboarding process.")
        
        # Add 6 text boxes to get user information
        user_full_name = st.text_input("Full Name")
        user_contact_number = st.text_input("Your contact Number")
        doctor_name = st.text_input("Doctor's Name")
        doctor_email = st.text_input("Doctor's Email")
        sos_contact_name = st.text_input("SOS Contact Name")
        sos_contact_number = st.text_input("SOS Contact Number")
        
        # Button to trigger the function
        if st.button("Submit"):
            save_onboarding_info(user_full_name, doctor_name, doctor_email, sos_contact_name, sos_contact_number, user_contact_number)
            st.write("All details saved.")
            print("All details saved.")
            st.session_state.page = "confirmation"  # Set page to confirmation after submitting the form
            st.rerun()  # Refresh the page to show confirmation page

    # Confirmation page
    elif st.session_state.page == "confirmation":
        st.write("All details saved.")
        
        # Button to go back to the form page for editing details
        if st.button("Edit Details"):
            st.session_state.page = "form"
            st.rerun()  # Refresh the page to go back to the form page


# Chat Tab
with tabs[0]:
    all_stm_summary = ""
    #invoke_llm(f'''Summarize this text in 5-6 lines. make sure to include all important points: {current_LTM}''')
    stm = ""

    # Streamlit UI
    st.title("MindMend : Lets talk!")

    # Initialize session state for storing chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Input field for user question
    question = st.text_input("Enter your question:")

    if "all_stm_summary" not in st.session_state:
        st.session_state.all_stm_summary = ""

    # Handle button press
    if st.button("Send"):
        if question.strip():
            with st.spinner("Processing..."):
                start_time = time.time()

                # Get the response (replace with your RAG model call)
                response = perform_RAG(question)
                st.text_area("Answer", value=response, height=200, disabled=True) 
                # TODO : improve prompt!!!
                stm = invoke_llm(f'''summrize this in 1-2 lines. make sure to capture the 
                                        sentiment and emotion of this chat: 
                                        "role": "user", "content": {question}
                                        "role": "chatbot", "content": {response}''')
                print(stm)
                # TODO : improve prompt!!!
                st.session_state.all_stm_summary += stm
                
                # print(all_stm_summary)

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


            # TODO : For each questions see if its an emergency : SOS calling API

            # TODO : improve prompt!!!


    if st.button('End Chat'):
        # TODO: Ask finally are you Okay, if not: call SOS, add in email that patient wasnt okay even after the chat.
        # udpate LTM in file
        # TODO : improve prompt!!!
        
        print("----------------------------------------------------------")
        print(st.session_state.all_stm_summary)
        print("----------------------------------------------------------")

        new_LTM = invoke_llm(f'''summarize in 20 lines: {st.session_state.all_stm_summary + current_LTM}''')
        
        with open(LTM_file_path, 'w') as file:
            file.write(new_LTM)
        # send STM to doctor
        email_body = st.session_state.all_stm_summary
        send_email(email_body)
        emergency_calling()
        
        # TODO : Strip thr summary from all_stm_summary etdc or tell llm to return on summary and nothing else
        