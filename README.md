# MindMend: Chatbot for Mental Health Support

## Overview

**MindMend** is a chatbot designed to support individuals facing mental health challenges. This application provides users with personalized interactions that reflect the principles and perspectives of their specific therapist, ensuring timely and relevant assistance.

## Problem Statement

In the USA, nearly 57% of individuals experience mental health issues at some point in their lives. Stigma and fear often prevent open discussions about these challenges, leading to reluctance in seeking therapy. MindMend addresses the critical issue of therapist unavailability by offering a continuous conversational support system that aligns with the guidelines and thinking processes of individual therapists. This way, users can access support whenever needed.

## Project Objectives

- Develop a chatbot capable of engaging in meaningful conversations with users through a user-friendly interface.
- Implement Retrieval-Augmented Generation (RAG) to provide responses that align with the specific therapeutic approach of a user’s designated therapist.
- Ensure the chatbot avoids triggering anxiety, depression, or suicidal thoughts.
- Enable the chatbot to summarize conversations and create notes for users to discuss with their therapists during the next appointment.

## Methodology

### Approach

MindMend serves as a supportive tool for patients while maintaining the therapist's role. 

### LLMs and Techniques

- **Model Selection**: The chatbot utilizes Retrieval-Augmented Generation to contextualize responses, focusing on Llama 3.2, GPT-4, and Gemini.
- **Frameworks**: Flask is used to create the user interface, with deployment options on Heroku and potential local execution for enhanced privacy.

### Architecture/Process

1. **Data Collection**: Multimodal data is gathered, including text (blogs), audio (podcasts), and video (YouTube).
2. **Data Preprocessing**: All modalities are converted to text format.
3. **Chatbot Development**: A basic chatbot is built using the selected large language model.
4. **Integration**: RAG is applied to enhance response quality.
5. **User Interface**: A functional UI is developed for user engagement.
6. **Deployment**: The chatbot is deployed on Heroku or can be run locally.

### Data Sources

Data is sourced from publicly available materials from mental health NGOs, including blogs, podcasts, and videos. Non-text data is transcribed and preprocessed to ensure comprehensive text data for training.

### Evaluation

Success metrics include accuracy and F1 score to assess the performance and reliability of the chatbot's responses.

## Related Work

### Existing Systems

- **Woebot Health**: Engages users in conversations about mental health, offering doctor referrals but acknowledging limitations in reliability.
- **Wysa**: A conversational AI that provides activities and suggestions, functioning similarly to a supportive resource.

### Positioning

MindMend uniquely personalizes responses according to individual therapists, addressing the inconsistencies that may arise when patients receive conflicting advice from different sources. It serves to supplement, not replace, professional help, recognizing the limitations of AI in handling severe mental health crises.

## Challenges and Risks

- **Doctor Replacement Risk**: It is crucial to emphasize that MindMend is not intended to replace therapists. The chatbot prompts users to consult their doctor as needed.
- **Data Requirements**: Ensuring the chatbot is effective requires a substantial amount of data, which is sourced from reputable organizations.
- **Security and Privacy**: All conversations must remain confidential. Deployment on secure platforms or local systems is prioritized to protect user data.

## Resources Needed

- **Hardware/Software**: A vector database, a deployment platform (e.g., Heroku), or a capable local machine for model execution.
- **Data Requirement**: Access to publicly available blogs, YouTube videos, and podcasts related to mental health.

## Conclusion

MindMend stands as an innovative solution to provide mental health support when therapists are unavailable, ensuring users receive consistent, personalized guidance aligned with their therapeutic relationship. By fostering a safe and supportive environment, MindMend aims to make a positive impact on mental health management.
