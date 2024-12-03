import os, re, json, time
from pydantic import BaseModel, Field
import openai
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime


ollama_client = openai.Client(base_url="http://127.0.0.1:11434/v1", api_key="EMPTY")
stop = ['Observation:', 'Observation ']

doctor_email_file_path = "onboarding-details/doctors-email.txt"
user_fullname_path = "onboarding-details/user-full-name.txt"

with open(user_fullname_path, 'r') as file:
    user_full_name = file.read()

password_path = "confidential/email_pass.txt"
# Open and read the file for email password
with open(password_path, 'r') as file:
    passkey = file.read()  # Read the entire content of the file 


def invoke_llm(prompt:str) -> str:
    try:
        response = ollama_client.completions.create(
            model="llama3.1",
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

def send_email():
    to_addr = get_doctors_email()
    now = datetime.now()
    formatted_date_time = now.strftime("%Y-%m-%d %H:%M:%S")
    print("user name is: ", user_full_name)
    print("Doctors email is: ", to_addr)
    subject = f'''Summary of conversation with {user_full_name} on {formatted_date_time}'''
    body = "abc for now" ##### chagne this

    output = send_email_internal(to_addr, subject, body)
    print(output)
    # try:
    #     patch_json_content = json.loads(llm_json_str)
    #     to_addr = patch_json_content["to_addr"]
    #     subject = patch_json_content["subject"]
    #     body = patch_json_content["body"]

    # except Exception as e:
    #     error_str = f"Exception: {e}"

    # output = send_email_internal(to_addr, subject, body)
    # return output

send_email()
