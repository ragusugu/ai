from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Optional
import requests
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

APP_URL = "http://app:8000/ask"
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemma4:12b")

class Prompt(BaseModel):
    message: str
    model: str | None = DEFAULT_MODEL
    history: Optional[List[Dict[str, str]]] = []

@app.post("/chat")
def chat(prompt: Prompt):
    payload = {
        "prompt": prompt.message,
        "history": prompt.history
    }
    try:
        r = requests.post(APP_URL, json=payload, stream=True, timeout=120)
        r.raise_for_status()
        def generate():
            for chunk in r.iter_content(chunk_size=None):
                if chunk:
                    yield chunk
        return StreamingResponse(generate(), media_type="application/x-ndjson")
    except requests.RequestException as e:
        return {"response": f"Error: {str(e)}"}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    files = {"file": (file.filename, file.file, file.content_type)}
    try:
        r = requests.post(APP_URL.replace("/ask", "/upload"), files=files)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        return {"status": "error", "message": str(e)}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/", StaticFiles(directory="ui", html=True), name="ui")
