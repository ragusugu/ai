from fastapi import FastAPI
from pydantic import BaseModel
import requests
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

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
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    try:
        r.raise_for_status()
    except requests.RequestException as e:
        return {"error": str(e)}
    return r.json()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://localhost:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/", StaticFiles(directory="ui", html=True), name="ui")
