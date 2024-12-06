# MindMend: Chatbot for Mental Health Support

## Overview

**MindMend** 
mental health chatbot that assists users needing therapy for mental well-being.

## Problem Statement

In the USA, nearly 57% of individuals experience mental health issues at some point in their lives. Stigma and fear often prevent open discussions about these challenges, leading to reluctance in seeking therapy. MindMend addresses the critical issue of therapist unavailability by offering a continuous conversational support system that aligns with the guidelines and thinking processes of individual therapists. This way, users can access support whenever needed.

<img width="989" alt="image" src="https://github.com/user-attachments/assets/55076060-5811-4291-9d36-cae517d74e94">


## Project Features

- Create user profile (all information saved locally so secure): Name, contact, doctor's name, email and most importantly: SOS contact
- Provide automatic emergency calliing to SOS if self harming tendency is seen in chats.
- Answer in text as well as audio 
- Enable the chatbot to summarize conversations and create notes that are automatically sent to the doctor via email once chat ends
- Maintain context in a single chat and also across different chats.

## Technical details
Large Language Model used: Llama 3.2: 3B model
Calling API: Twillio
Email Protocol: SMTP

## Using the bot:

- Clone the repository:  
```bash
git clone https://github.com/nehayj100/MindMend-Mental-Health-Chatbot-Assistant
cd Pipeline
```
- Install required libraries:  
```bash
pip install -r requirements.txt
```
- Run the application to land on UI
```bash
    streamlit run MindMend.py
```

Demo:

<img width="385" alt="image" src="https://github.com/user-attachments/assets/8b6dcf88-12dc-46cf-b80b-dc9d8c77086a">

<img width="383" alt="image" src="https://github.com/user-attachments/assets/0cba0870-7c99-45f5-9045-254a63cb3393">

<img width="417" alt="image" src="https://github.com/user-attachments/assets/198ce86a-0472-49f7-870f-d6f3e4062e06">




