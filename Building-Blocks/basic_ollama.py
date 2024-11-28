import openai
client = openai.Client(
    base_url="http://127.0.0.1:11434/v1", api_key="EMPTY")

# Text completion
response = client.completions.create(
    model="llama3.1",
    prompt="The capital of France is",
    temperature=0,
    max_tokens=32,
)
print(f"answer: {response.choices[0].text}")