import os
import re, json, time
from pydantic import BaseModel, Field
import openai
import logging
import streamlit as st
import openai
import ollama
from gtts import gTTS
from dotenv import load_dotenv
import shelve
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
    return

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

load_dotenv()
USER_AVATAR = "👤"
BOT_AVATAR = "🤖"



# Load chat history from shelve file
def load_chat_history():
    with shelve.open("chat_history") as db:
        return db.get("messages", [])


# Save chat history to shelve file
def save_chat_history(messages):
    with shelve.open("chat_history") as db:
        db["messages"] = messages


# Initialize or load chat history
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

# Set the page config as the first Streamlit command
#st.set_page_config(page_title="MindMend : Let's talk!", layout="wide")

# Sidebar for navigation and chat history deletion
with st.sidebar:
    # Button to delete chat history
    st.title("MindMend")
    if st.button("Delete Chat History"):
        st.session_state.messages = []
        save_chat_history([])

    # Navigation buttons for tabs
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Home"  # Default to the "Chat" tab

    if st.button("Home"):
        st.session_state.active_tab = "Home"
    if st.button("Onboarding"):
        st.session_state.active_tab = "Onboarding"
    if st.button("Chat"):
        st.session_state.active_tab = "Chat"
    if st.button("End Chat"):
        st.session_state.active_tab = "EndChat"
if "all_stm_summary" not in st.session_state:
        st.session_state.all_stm_summary = ""
# Display content based on the active tab
if st.session_state.active_tab == "Chat":
    st.title("Lets talk!")
    for message in st.session_state.messages:
        avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # Chat Tab
    all_stm_summary = ""
    #invoke_llm(f'''Summarize this text in 5-6 lines. make sure to include all important points: {current_LTM}''')
    stm = ""

    # Initialize session state for storing chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []


    if "all_stm_summary" not in st.session_state:
        st.session_state.all_stm_summary = ""

    # Handle button press
    if prompt := st.chat_input("How can I help?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)
        start_time = time.time()
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            message_placeholder = st.empty()
            full_response = perform_RAG(prompt)
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        stm = invoke_llm(f'''summrize this in 1-2 lines. make sure to capture the 
                                    sentiment and emotion of this chat: 
                                    "role": "user", "content": {prompt}
                                    "role": "chatbot", "content": {full_response}''')
        
        st.session_state.all_stm_summary += stm

        # Convert the AI response to audio and play it
        audio_path = text_to_speech(full_response)
        audio_file = open(audio_path, "rb")
        st.audio(audio_file.read(), format="audio/mp3")

        os.remove(audio_path)

        end_time = time.time()
        print("Total time: ", end_time - start_time)

            # TODO : For each questions see if its an emergency : SOS calling API

            # TODO : improve prompt!!!
    save_chat_history(st.session_state.messages)

elif st.session_state.active_tab == "EndChat":
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
    st.session_state.active_tab = "Home"
    st.rerun()
    # TODO : Strip thr summary from all_stm_summary etdc or tell llm to return on summary and nothing else
    
elif st.session_state.active_tab == "Home":
    st.title("Welcome to MindMend")
    st.markdown("""
        MindMend is here to support your mental health journey. We understand that seeking help and maintaining your mental well-being is important. Our chatbot is available to chat with you anytime you need.

        ### How to use MindMend:
        1. **Onboarding**: Understand how to use the chatbot and set up your preferences.
        2. **Chat**: Start a conversation with the chatbot for emotional support, advice, or coping strategies.
        3. **End Chat**: End your session when you're ready.

        If you ever feel overwhelmed, remember that you are not alone, and support is available. Take care of your mental health.

        ### Quick Resources:
        - **Crisis Helplines**: If you're in immediate need of help, check out local crisis helplines or chat services.
        - **Mindfulness Tips**: Practice breathing exercises or mindfulness activities to relax.
        - **Mental Health Articles**: Learn about managing stress, anxiety, and other mental health topics.

        Stay calm, stay strong, and reach out whenever you need assistance.
    """)

    st.markdown("#### Ready to start? Select one of the tabs from the sidebar to begin your journey.")
    st.markdown("""
        - **Onboarding**: Set your preferences and get acquainted with the features of MindMend.
        - **Chat**: Start a supportive chat with the bot.
        - **End Chat**: End the session when you're ready.
    """)

elif st.session_state.active_tab == "Onboarding":
    if "page" not in st.session_state:
        st.session_state.page = "form"  # Default to the form page

    # Form page for user input
    if st.session_state.page == "form":
        # st.title("Welcome to MindMend!")
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
