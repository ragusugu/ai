from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Optional
from fastapi.responses import StreamingResponse
import requests
import json
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
import pathlib
import os
import shutil
import uuid
from app.ingest_docs import read_file, get_chunks, collection as ingest_collection, model as ingest_model, client as ingest_client

OLLAMA_URL = "http://ollama:11434/api/generate"

app = FastAPI(title="Private AI Assistant")

# Load system prompt
system_prompt_path = pathlib.Path(__file__).parent / "system_prompt.txt"
with open(system_prompt_path) as f:
    SYSTEM_PROMPT = f.read()

# Load memory
def load_memory():
    try:
        with open("/data/memory.json") as f:
            return json.load(f)
    except:
        return {}

memory = load_memory()

# Vector DB and Reranker
embedder = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
client = chromadb.Client(
    chromadb.config.Settings(persist_directory="/data/vectordb")
)
collection = client.get_or_create_collection("personal_knowledge")

class Query(BaseModel):
    prompt: str
    history: Optional[List[Dict[str, str]]] = []

def retrieve_context(query, k=4, fetch_k=15):
    if collection.count() == 0:
        return ""
    emb = embedder.encode(query).tolist()
    
    # Fetch more documents for reranking
    res = collection.query(query_embeddings=[emb], n_results=fetch_k)
    docs = res["documents"][0]
    
    if len(docs) <= k:
        return "\n\n".join(docs)
        
    # Reranking with CrossEncoder
    pairs = [[query, doc] for doc in docs]
    scores = reranker.predict(pairs)
    
    scored_docs = list(zip(scores, docs))
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    top_docs = [doc for score, doc in scored_docs[:k]]
    return "\n\n".join(top_docs)

@app.post("/ask")
def ask_ai(query: Query):
    context = retrieve_context(query.prompt)

    profile = f"""
User Profile:
Name: {memory.get("name")}
Role: {memory.get("role")}
Skills: {", ".join(memory.get("skills", []))}
"""

    skills = memory.get("skills") or []
    if isinstance(skills, str):
        skills = [skills]
    skills_str = ", ".join(skills)

    history_str = ""
    if query.history:
        for msg in query.history:
            role = msg.get("role", "User").capitalize()
            content = msg.get("content", "")
            history_str += f"{role}: {content}\n"

    full_prompt = f"""
{SYSTEM_PROMPT}

{profile}

Relevant knowledge:
{context}

History:
{history_str}

User: {query.prompt}
Assistant:
"""

    payload = {
        "model": os.getenv("MODEL_NAME", "llama3"),
        "prompt": full_prompt,
        "stream": True
    }

    try:
        r = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=120)
        r.raise_for_status()
        def generate():
            for chunk in r.iter_content(chunk_size=None):
                if chunk:
                    yield chunk
        return StreamingResponse(generate(), media_type="application/x-ndjson")
    except requests.RequestException as e:
        return {"answer": f"Error communicating with AI model: {str(e)}"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    docs_dir = "/data/docs"
    os.makedirs(docs_dir, exist_ok=True)
    
    file_path = os.path.join(docs_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    text = read_file(file_path)
    if not text.strip():
        return {"status": "error", "message": "Could not extract text from file"}
        
    for i, ch in enumerate(get_chunks(text)):
        embedding = ingest_model.encode(ch).tolist()
        ingest_collection.add(
            documents=[ch],
            embeddings=[embedding],
            ids=[f"{file.filename}_{i}_{uuid.uuid4().hex}"]
        )
    
    ingest_client.persist()
    return {"status": "success", "message": f"Successfully ingested {file.filename}"}
