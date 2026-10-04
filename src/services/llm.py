import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

MODEL = "llama3.2"


def ask_llm(question, context):

    prompt = f"""
You are a helpful AI assistant.

Answer ONLY from the given context.

If answer is not present in context,
say "Answer not found in document."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        }
    )

    data = response.json()

    return data["response"]