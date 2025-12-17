# Copilot / Agent Instructions for Private AI Assistant

This file captures the essential, discoverable knowledge an AI coding agent needs to be productive in this repo.

- **Architecture (big picture):** The project runs a local LLM host (`ollama`) and two small FastAPI services:
  - `ollama` (container) — model host exposed at `http://ollama:11434`.
  - `backend/` — lightweight FastAPI proxy that exposes `POST /chat` and forwards to Ollama (used by the web UI).
  - `app/` — assistant service with richer behavior: `/ask` composes a system prompt, user profile (from `/data/memory.json`), retrieves contextual docs from a local ChromaDB vector store and queries Ollama.

- **Key files to read first:** `app/main.py`, `app/ingest_docs.py`, `app/system_prompt.txt`, `backend/app.py`, `docker-compose.yml`, `Dockerfile`, `ReadMe.txt`.

- **Data locations & persistence:**
  - Persistent data expected at `/data` inside containers: `/data/memory.json`, `/data/docs` (source docs for indexing), `/data/vectordb` (ChromaDB persist dir).
  - If you run containers, mount a host `./data` into `/data` (e.g. `-v $(pwd)/data:/data`) so memory and vectordb persist across restarts.

- **Vector DB & embeddings:**
  - `app/ingest_docs.py` indexes `PDF`, `DOCX`, `TXT`, `JSON` from `/data/docs` into a Chroma collection named `personal_knowledge`.
  - Embeddings use `SentenceTransformer("all-MiniLM-L6-v2")` and `chunk()` size is ~500 words.
  - The code persists the Chroma DB to `persist_directory="/data/vectordb"`.

- **Prompt composition & memory:**
  - `app/main.py` builds the assistant prompt by concatenating `app/system_prompt.txt`, a `User Profile` derived from `/data/memory.json` (keys: `name`, `role`, `skills`), relevant docs from the vector DB, and the user query.
  - `system_prompt.txt` includes an explicit rule: "Never send data outside this machine" — treat it as a functional constraint.

- **Model / Ollama integration:**
  - Both services post to `http://ollama:11434/api/generate`.
  - `app/main.py` requests model `llama3`; `backend/app.py` defaults to `phi3:mini` (see `DEFAULT_MODEL`).
  - To pre-load models in the `ollama` container: `docker exec -it ollama ollama pull <model>` (e.g. `llama3` or `phi3:mini`).

- **Run / dev workflows:**
  - Preferred (containerized): `docker compose up -d` (see `ReadMe.txt`).
  - Rebuild after changing `Dockerfile`, `docker-compose.yml`, or `requirements.txt`:
    - `docker compose down`
    - `docker compose up -d --build`
  - Local (no Docker) quick run for `app/`:
    - `pip install -r requirements.txt`
    - `uvicorn app.main:app --reload --port 8000`
    - For `backend/` (separate process): `uvicorn app:app --reload --port 8000` from within `backend/`.

- **HTTP endpoints to test:**
  - Assistant (app): `POST /ask` expects JSON `{ "prompt": "..." }` — example `curl` in `ReadMe.txt`.
  - Backend (proxy): `POST /chat` expects `{ "message": "...", "model": "optional" }` and returns raw Ollama JSON.
  - Web UI (`ui/index.html`) calls `/chat` and expects a JSON `response` string.

- **Project-specific conventions & gotchas:**
  - `memory.json` shape is assumed (top-level keys used directly). If missing, `app/main.py` falls back to an empty memory dict.
  - The vector retrieval in `app/main.py` returns joined `documents`; watch ordering and length when constructing prompts.
  - Timeouts: `app/main.py` passes `timeout=120` to request; long-running model calls can take time.
  - There are two FastAPI apps with slightly different behaviors; be careful which service you change — the UI points to `backend`.

- **Quick references (commands):**
  - Start everything: `docker compose up -d`
  - Stop: `docker compose down`
  - Rebuild: `docker compose up -d --build`
  - Pull model into Ollama: `docker exec -it ollama ollama pull llama3` or `docker exec -it ollama ollama pull phi3:mini`
  - Index docs: run `python app/ingest_docs.py` inside a container or host with `/data/docs` mounted.

- **When editing code related to models/data:**
  - If you change embedding model, chunk size, or Chroma settings, re-run `ingest_docs.py` to re-index.
  - If you change `system_prompt.txt` or `memory.json`, tests or manual curl checks are useful to validate prompt composition.

If anything in this file is unclear or you want more detail (examples of `memory.json`, sample `docker-compose` mounts, or CI/test commands), say what you'd like clarified and I'll iterate.
