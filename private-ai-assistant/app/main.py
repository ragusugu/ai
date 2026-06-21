from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import os
import pathlib
import requests
import shutil
import uuid
from sentence_transformers import CrossEncoder, SentenceTransformer

from app.ingest_docs import (
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE,
    client as ingest_client,
    collection,
    get_chunks,
    model as ingest_model,
    read_file,
)
from app.memory_store import MemoryStore

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434/api/generate")
MEMORY_PATH = os.getenv("MEMORY_PATH", "/data/memory.json")
DOCS_DIR = os.getenv("DOCS_DIR", "/data/docs")
DEFAULT_MODEL = os.getenv("MODEL_NAME", "llama3")

app = FastAPI(title="Private AI Assistant")
memory_store = MemoryStore(MEMORY_PATH)

system_prompt_path = pathlib.Path(__file__).parent / "system_prompt.txt"
with open(system_prompt_path, encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

embedder = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
class Query(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=20000)
    model: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = []


class MemoryFact(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    tags: List[str] = []


class ProfilePatch(BaseModel):
    profile: Dict[str, Any]


def safe_filename(filename: str) -> str:
    name = pathlib.PurePath(filename or "upload").name
    cleaned = "".join(c if c.isalnum() or c in {".", "-", "_"} else "_" for c in name)
    if not cleaned or cleaned in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid filename")
    ext = pathlib.Path(cleaned).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
    return cleaned


def retrieve_context(query: str, k: int = 4, fetch_k: int = 15) -> str:
    if collection.count() == 0:
        return ""
    emb = embedder.encode(query).tolist()
    res = collection.query(query_embeddings=[emb], n_results=fetch_k)
    docs = res.get("documents", [[]])[0]
    if len(docs) <= k:
        return "\n\n".join(docs)

    pairs = [[query, doc] for doc in docs]
    scores = reranker.predict(pairs)
    scored_docs = list(zip(scores, docs))
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return "\n\n".join(doc for _, doc in scored_docs[:k])


def format_history(history: Optional[List[Dict[str, str]]]) -> str:
    if not history:
        return ""
    lines = []
    for msg in history[-12:]:
        role = (msg.get("role") or "user").capitalize()
        content = msg.get("content") or ""
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/memory")
def get_memory() -> dict[str, Any]:
    return memory_store.load()


@app.patch("/memory/profile")
def update_profile(payload: ProfilePatch) -> dict[str, Any]:
    return memory_store.update_profile(payload.profile)


@app.post("/memory/facts")
def add_memory_fact(payload: MemoryFact) -> dict[str, Any]:
    return memory_store.add_fact(payload.text, payload.tags)


@app.post("/ask")
def ask_ai(query: Query):
    context = retrieve_context(query.prompt)
    model_name = query.model or DEFAULT_MODEL
    full_prompt = f"""
{SYSTEM_PROMPT}

{memory_store.prompt_context()}

Relevant knowledge:
{context}

Current chat history:
{format_history(query.history)}

User: {query.prompt}
Assistant:
"""

    payload = {"model": model_name, "prompt": full_prompt, "stream": True}

    try:
        r = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=120)
        r.raise_for_status()

        def generate():
            chunks: list[str] = []
            for chunk in r.iter_content(chunk_size=None):
                if chunk:
                    text = chunk.decode("utf-8", errors="ignore")
                    chunks.append(text)
                    yield chunk
            assistant_text = _extract_ollama_response("".join(chunks))
            if assistant_text.strip():
                memory_store.add_conversation(query.prompt, assistant_text, model_name)

        return StreamingResponse(generate(), media_type="application/x-ndjson")
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Error communicating with AI model: {str(e)}") from e


def _extract_ollama_response(ndjson_text: str) -> str:
    parts: list[str] = []
    for line in ndjson_text.splitlines():
        if not line.strip():
            continue
        try:
            import json

            data = json.loads(line)
        except ValueError:
            continue
        if data.get("response"):
            parts.append(data["response"])
    return "".join(parts)


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    filename = safe_filename(file.filename)
    os.makedirs(DOCS_DIR, exist_ok=True)
    file_path = os.path.join(DOCS_DIR, f"{uuid.uuid4().hex}_{filename}")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if os.path.getsize(file_path) > MAX_FILE_SIZE:
        os.remove(file_path)
        raise HTTPException(status_code=413, detail="File is too large")

    text = read_file(file_path)
    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from file")

    chunks = get_chunks(text)
    for i, ch in enumerate(chunks):
        embedding = ingest_model.encode(ch).tolist()
        collection.add(
            documents=[ch],
            embeddings=[embedding],
            metadatas=[{"filename": filename, "chunk": i, "source_path": file_path}],
            ids=[f"{filename}_{i}_{uuid.uuid4().hex}"],
        )

    if hasattr(ingest_client, "persist"):
        ingest_client.persist()
    memory_store.add_fact(f"Uploaded and indexed document: {filename}", ["document", "upload"])
    return {"status": "success", "message": f"Successfully ingested {filename}", "chunks": len(chunks)}
