from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI()

OLLAMA_URL = "http://ollama:11434/api/generate"
DEFAULT_MODEL = "phi3:mini"

class Prompt(BaseModel):
    message: str
    model: str | None = DEFAULT_MODEL

@app.post("/chat")
def chat(prompt: Prompt):
    payload = {
        "model": prompt.model,
        "prompt": prompt.message,
        "stream": False
    }
    r = requests.post(OLLAMA_URL, json=payload)
    return r.json()
