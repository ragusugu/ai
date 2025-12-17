from fastapi import FastAPI
from pydantic import BaseModel
import requests, json
import chromadb
from sentence_transformers import SentenceTransformer

OLLAMA_URL = "http://ollama:11434/api/generate"

app = FastAPI(title="Private AI Assistant")

# Load system prompt
with open("app/system_prompt.txt") as f:
    SYSTEM_PROMPT = f.read()

# Load memory
def load_memory():
    try:
        with open("/data/memory.json") as f:
            return json.load(f)
    except:
        return {}

memory = load_memory()

# Vector DB
embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.Client(
    chromadb.config.Settings(persist_directory="/data/vectordb")
)
collection = client.get_or_create_collection("personal_knowledge")

class Query(BaseModel):
    prompt: str

def retrieve_context(query, k=4):
    if collection.count() == 0:
        return ""
    emb = embedder.encode(query).tolist()
    res = collection.query(query_embeddings=[emb], n_results=k)
    return "\n".join(res["documents"][0])

@app.post("/ask")
def ask_ai(query: Query):
    context = retrieve_context(query.prompt)

    profile = f"""
User Profile:
Name: {memory.get("name")}
Role: {memory.get("role")}
Skills: {", ".join(memory.get("skills", []))}
"""

    full_prompt = f"""
{SYSTEM_PROMPT}

{profile}

Relevant knowledge:
{context}

User: {query.prompt}
Assistant:
"""

    payload = {
        "model": "llama3",
        "prompt": full_prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()

    return {"answer": response.json()["response"].strip()}
