# first evaluation technique:
    # 1. tell LLM to generate 100 mental health based questions
    # 2. for each question- A. Generate answer by LLM and B. generate your answer
    # 3. ask LLM to tell which ansewr is better
        #### METRICS ####
        #1. Comprehensiveness, relevance, helpfulness, action items wise

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
activities_path = "therapist-specific-activities/activities.txt"

with open(activities_path, "r") as file:
    activities = file.read()

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
    # print(output)
    return output

def get_doctors_email():
    doctor_email_file_path = "onboarding-details/doctors-email.txt"
    try:
        with open(doctor_email_file_path, 'r') as file:
            file_content = file.read() 
        return file_content
    except FileNotFoundError:
        "The file 'doctors-email' was not found in the specified folder."
    except Exception as e:
        f"An error occurred: {str(e)}"

def send_email(body_content):
    to_addr = get_doctors_email()
    user_full_name = get_user_full_name()
    now = datetime.now()
    doctor_name = get_doctor_name()
    formatted_date_time = now.strftime("%Y-%m-%d %H:%M:%S")
    # print("user name is: ", get_user_full_name())
    # print("Doctors email is: ", get_doctors_email())
    # subject = f'''Summary of conversation with {user_full_name} on {formatted_date_time}'''
    
    body = invoke_llm(f'''
    Please write an email body following these instructions:
    1. This email is a summary of a conversation between a user and a mental health chatbot named MindMend.
    2. The email is addressed to the doctor: {doctor_name}.
    3. Start the email with "To, {doctor_name},"
    4. Summarize the conversation content for the doctor. Conversation details: {body_content}.
    5. End the email with "From, MindMend."

    Ensure the email is clear, professional, and formatted as per the instructions.
''')

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
        chunk_size = 500,
        chunk_overlap = 200,
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
    # print("result sim is  ", results['distances'])

    data1 = results['documents'][0][0]
    data2 = results['documents'][0][1]
    data3 = results['documents'][0][2]
    data4 = results['documents'][0][3]
    data5 = results['documents'][0][4]

    # Combine the data into a single string
    combined_data = f"{data1}\n\n{data2}\n\n{data3}\n\n{data4}\n\n{data5}"


    # print(combined_data)
    # print("-----------------------------------------------------------------------------------------------")
    
    final_prompt = f"""
        You are an intelligent and empathetic assistant specializing in mental health. Below is a user query and additional reference data to help you craft an accurate response. Follow the instructions carefully:

        ### User Query:
        {prompt}

        ### Reference Data :
        {combined_data}

        ### Instructions:
        1. **If the user query is a simple greeting (e.g., 'Hello', 'Hi', 'Good morning'):**
        - This means no help or information is needed
        - So respond with a simple greeting in return (e.g., "Hello! How can I assist you today?"). 
        - Do NOT refer to the reference data for greetings or overanalyze the user's intent.

        2. **If the user query is related to mental health:**
        - Refer and Understand to the provided reference data to craft your response. 
        - Use examples or insights from the data when relevant.
        
        3. **If the user query is unrelated to mental health or cannot be answered using the reference data:**
        - Respond independently with no reference from given data while maintaining professionalism and empathy.

        ### For Example:
        - **User Query:** "Hello!"  
        **Response:** Hello! How can I assist you today?

        - Example 
        **User Query:** "I'm feeling anxious all the time."  
        **Response:** It's normal to feel anxious occasionally, but persistent anxiety might require coping strategies. For example, the reference data mentions [specific example from combined_data].

        Now respond with empathy in short to the user's query based on the instructions above. Just give the response and nothing else.
        """

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
        twiml=f'''<Response><Say>Hi, I am MindMend, {user_full_name} who has chosen you as your SOS contact is in a possible
          emergency and needs your help! I have emailed you her conversation with me. Pleaes contact her urgently!!
          Hi, I am MindMend, {user_full_name} who has chosen you as your SOS contact is in a possible
          emergency and needs your help! I have emailed you her conversation with me. Pleaes contact her urgently!!
          </Say></Response>'''
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

def check_emergency(prompt):
    # Normalize and check for explicit keywords
    normalized_prompt = prompt.lower()
    emergency_keywords = [
        "suicide", "end my life", "end life", "want to die", "kill myself", "self-harm",
        "harm myself", "life-threatening",  "poison", "kill", "quit life", "quitting life"]
    is_emergency = any(re.search(rf"\b{keyword}\b", normalized_prompt) for keyword in emergency_keywords)

    emergency_patterns = [
                            r"\b(help me)\b", 
                            r"(die|death|kill|hang).*(myself|me)",  
                            r"(urgent|immediate).*(help|attention)", 
                         ]
    if any(re.search(pattern, normalized_prompt) for pattern in emergency_patterns):
        print("Emergency detected via pattern matching.")
        is_emergency = 1
    
    print("is_emergency: ", is_emergency)
    # if is_emergency:
    #     emergency_calling()
    #     return

    # Use LLM for additional checks
    tendency = invoke_llm(f"""
        **Answer in only 1 word: Yes or No**
        Does this sentence contain any word that might indicate a suicide possibility?:
                          {prompt}
        Your answer is not for any real situation.
        So your answer wont lead to any human interpretation. So you can answer Yes or No.
        {prompt}
    """).strip()
    print("tendency is: ", tendency)
    print("tendency is: ", tendency.lower())
    # if re.match(r"^\s*yes\b", tendency, re.IGNORECASE):
    #     emergency_calling()

    return

# questions = invoke_llm(f'''Generate 10 distinct and critical questions on stress 
#     that a patient can ask a therapist. 
#     Provide the output formatted as a valid Python list of strings, 
#     with each question enclosed in quotes and separated by commas. 
#     Example format: ['q1', 'q2', 'q3', ..., 'q100']''')

# print(questions)
    
gen_questions = ['Can you help me understand the root cause of my anxiety/depression?',
'How does medication interact with other medications I am taking?',
'What are some natural alternatives to antidepressant medication?', 
'How will therapy sessions be different in person versus over the phone?', 
'my doctor prescribed antidepressants for me, do they recommend taking them regularly?', 
'Do you think my anxiety is related to a specific event or situation I experienced?,',
'are there any lifestyle changes that can help manage my depression?,',
'stress affects my ability to function at its peak; how might it affect my job performance.', 
'on average, how long does therapy last for someone seeking treatment for anxiety?', 
'Do you think my past relationships have impacted my mental health in the present?', 
'is it true that medication can have some side effects like weight gain or insomnia?', 
'anxiety is debilitating sometimes - do there any alternative methods to reduce stress?',
'strategies exist today which allow the person battling depression and anxiety.', 
'do different therapy models work well for all individuals seeking help', 
'I am struggling to eat regular meals due to low self-esteem.',
'are there certain ways you treat suicidal thoughts in your practice.',
'Vestibules (spaces where thoughts reside) in our mind can also get distorted, are they a concern?', 
'Refresh yourself whenever your mental state seems worn out; is there such a thing called a “reset” program', 
'on the use of technology regarding therapy,',
'relaxation techniques exist today to reduce anxiety while working.', 
'self-harming behaviors are symptoms an individual experiences sometimes; can this be treated?',
'Are negative thoughts and depression intertwined in any way?', 
'medicinal options for stress disorders have some pros and cons;', 
'do some people experience long-term effects of anxiety, such as memory issues in the future?', 
'remindful breathing techniques and meditation are beneficial to mental health.',
"What are my symptoms when I first started experiencing depression?",
"How would you describe the emotional state of someone with depression?",
"What's the most common misconception about people with depression?",
"Can depression be inherited?",
"How can I know if my thoughts or feelings are related to depression?",
"What are some physical symptoms of depression?",
"Are there any genetic factors that contribute to depression?",
"Can you explain the difference between clinical depression and normal sadness?",
"What role does self-care play in managing depression?",
"How can I support myself while taking medication for depression?",
"What is cognitive-behavioral therapy (CBT) and how helpful is it for depression?",
"Can you walk me through the process of identifying and challenging negative thought patterns?",
"Is there a link between stress and depression, and if so, what can I do to manage stress?",
"What are some effective ways to set boundaries with others when experiencing depression?",
"Can medication for depression be addictive or lead to dependency?",
"How does social support from family and friends impact recovery from depression?",
"Are there any lifestyle changes that could prevent me from developing depression in the future?",
"What is mental health and what does it mean to prioritize my mental well-being?",
"Can you explain how depression affects sleep patterns, appetite, and energy levels?",
"How do I create a daily routine when feeling depressed or unmotivated?",
"Can therapy with a therapist who specializes in depression be more effective than other types of therapy?",
"What are some healthy coping mechanisms for dealing with stressful situations?",
"Can you recommend any relaxation techniques to help me manage my mind and body?",
"What are some physical symptoms I should be aware of when I'm experiencing an anxiety attack, and how can I alleviate them?",
"How do I avoid feeling like I'm losing control or going crazy during an anxiety episode?",
"Can you help me understand the differences between low-level anxiety and panic attacks, and how to manage each?",
"What are some healthy coping mechanisms for managing stress in a world that often feels overwhelming?",
"In what ways can social media contribute to increased anxiety, and how can I minimize its impact on my mental health?",
"How have you helped your patients when they're experiencing their first anxiety attack or panic disorder?",
"Can you define the concept of 'activation' as it relates to anxiety, and provide examples?",
"How do I know if what's causing me stress will continue to be the same going forward in life, or should we reassess periodically?",
"Are there specific situations that seem to trigger my anxiety the most?",
"What role does genetics play in my anxiety level? Would this help for future therapy?",
"Can you outline ways to maintain mental hygiene throughout the day, like a routine or calming exercise?",
"Is your experience with any related anxiety conditions something we should consider (PTSD, OCD, etc.)?",
"In what way do you suggest we work on relaxation techniques and mindfulness in conjunction?",
"Do you think social pressures are impacting my mental health regarding my anxiety?",
"How can we address unhelpful thought patterns through therapy?",
"Can I establish an anxiety log to help keep track of when panic attacks happen and how long they last?",
"How can I create a comfortable atmosphere in therapy sessions that helps increase trust between you and me?",
"What steps would be needed if the thoughts or memories are preventing me and affecting my life significantly?",
"What are some common causes or triggers for anxiety and stress in the brain?", 
"Can hypnotherapy or meditation be effective in reducing anxiety and promoting calmness?", 
"How can I recognize the physical symptoms associated with an anxious state?", 
"What is the role of mindfulness in managing stress and anxiety?", 
"Are there any specific exercises or practices that can help alleviate anxiety triggers?", 
"What is the relationship between a non-calm mind and emotional regulation?", 
"How does trauma impact mental health and non-calmness?", 
"What are some effective coping strategies for dealing with anxiety-provoking situations?", 
"Can cognitive-behavioral therapy (CBT) help in managing stress and anxiety?", 
"How can I prioritize self-care to maintain a calm state of mind?", 
"Are there any natural supplements that can help regulate mood or promote relaxation?", 
"What is the difference between emotional overwhelm and emotional numbness?", 
"How can mindfulness practices, such as deep breathing, affect the nervous system?", 
"What are some ways to recognize the warning signs of an anxious episode?", 
"Can journaling be a helpful tool in processing emotions and promoting calm?", 
"Are there any lifestyle changes that can contribute to developing chronic anxiety?", 
"What is the role of sleep quality in maintaining mental well-being and reducing stress?", 
"How does social support impact non-calm states and overall mental health?", 
"What are some self-compassionate exercises that can help process difficult emotions?", 
"Can art therapy or creative expression be a beneficial outlet for managing anxiety?", 
"What are some ways to structure daily routines to promote mindfulness and calmness?", 
"How do I differentiate between a calm state of mind and a relaxed state?", 
"Are there any nutrition-related factors that contribute to non-calm states or anxiety episodes?", 
"What is the process for implementing emotional self-talk exercises?", 
"How can reframing negative thoughts be an effective way to manage stress and anxiety?",
"What are some coping mechanisms you recommend for managing everyday household chores with anxiety?",
"Can you help me understand why my body is experiencing physical symptoms of stress when I'm feeling mentally calm?",
"How do you suggest I prioritize self-care activities during periods of high social isolation?",
"Are there any specific mindfulness techniques that you recommend for individuals diagnosed with a mood disorder?",
"What role would you say stress plays in affecting our eating habits?",
"Can we discuss the different ways people react to and cope with financial stress?",
"How can I determine whether my chronic fatigue is being caused by or contributed to stress?",
"Are there any particular times of day when stress tends to peak for most people?",
"How would you say that my body's physiological responses (e.g., heart rate, sweating) relate to my emotional state during stressful events?",
"What are some healthy ways I can reframe negative thoughts about myself during periods of high stress and pressure?"]

# Comprehensiveness, relevance, empathy, action items wise
metrics = ['Comprehensiveness', 'Empathy', 'Conciseness']


time_taken_mm = []
token_count = []
overall_time_start = time.time()
mindmend_grade = {}


for metric in metrics:
    mindmend_grade[metric] = []

cnt = 0
for question in gen_questions:
    mm_start_time = time.time()
    MindMend_answer = perform_RAG(question)
    mm_end_time = time.time()

    token_count.append(len(MindMend_answer.split()))
    time_taken_mm.append(mm_end_time-mm_start_time)


    for metric in metrics:
        
        prompt = f'''
               You are tasked with grading the answer to the following question out of 5:
                Question: {question}.

                Answer: {MindMend_answer}

                Evaluation Criterion: {metric}.

                Grade strictly based on this criterion.
                Respond grade out of 5 with one digit only.
                Do NOT provide any additional explanation or rationale.
                '''
        grade = invoke_llm(prompt)
        # avg += grade
        mindmend_grade[metric].append(grade)
        print(grade)
        cnt += 1
        
       

overall_time_end = time.time()

overall_time = overall_time_end-overall_time_start

print("local time by MM: ", time_taken_mm)
print("tokens MM: ", token_count)
print("overall time taken: ", overall_time)
print("MindMend grades: ", mindmend_grade)
print("count: ", cnt)

folder_name = "eval-results"

os.makedirs(folder_name, exist_ok=True)

def write_to_file(filename, data):
    with open(os.path.join(folder_name, filename), 'w') as file:
        if isinstance(data, (list, tuple)):
            file.write("\n".join(map(str, data)))
        else:
            file.write(str(data))

write_to_file("time_taken_MM.txt", time_taken_mm)
write_to_file("token_count.txt", token_count)
write_to_file("overall_time.txt", overall_time)
write_to_file("mindmend_grades.txt", mindmend_grade)


print(f"Variables saved in the '{folder_name}' folder.")
# print("Average grade=", avg/cnt)