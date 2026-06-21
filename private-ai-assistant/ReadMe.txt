Private AI Assistant
====================

Local FastAPI + Ollama assistant with:
- Login-protected browser UI
- Streaming local chat
- Persistent JSON memory space
- Document upload and local RAG over PDF, DOCX, TXT, and JSON
- Chroma vector store persisted under ./data/vectordb

Project Layout
--------------

private-ai-assistant/
├── docker-compose.yml
├── Dockerfile                    # core RAG/Ollama API image
├── backend/Dockerfile            # browser UI/proxy/auth image
├── requirements.txt              # core app dependencies
├── backend/requirements.txt      # UI proxy dependencies
├── app/
│   ├── main.py                   # RAG API, upload, memory API
│   ├── memory_store.py           # durable JSON memory store
│   ├── ingest_docs.py            # document extraction/chunking
│   └── system_prompt.txt
├── backend/
│   └── app.py                    # login/session + proxy API
├── ui/
│   └── index.html                # self-contained browser UI
└── data/
    └── memory.json               # persistent memory profile/facts/conversations

First-Time Setup
----------------

1. Create local environment config:

   cp .env.example .env

2. Edit .env and change at least:

   AI_PASSWORD=your-local-password
   SESSION_SECRET=a-long-random-secret

3. Start services:

   docker compose up -d --build

4. Pull the model once:

   docker exec -it ollama ollama pull gemma4:12b

   If that model is not available on your Ollama install, use a local model you have, for example:

   docker exec -it ollama ollama pull llama3:8b-instruct-q4_K_M

   Then set MODEL_NAME=llama3:8b-instruct-q4_K_M in .env and rebuild/restart.

5. Open the UI:

   http://localhost:8000

6. Login using AI_USERNAME and AI_PASSWORD from .env.

Daily Workflow
--------------

Start:

  docker compose up -d

Stop:

  docker compose down

Rebuild after code, Dockerfile, Compose, or requirements changes:

  docker compose up -d --build

View logs:

  docker compose logs -f backend app ollama

Health checks:

  curl http://localhost:8000/health
  curl http://localhost:8001/health

Memory Workflow
---------------

Memory is stored locally in:

  ./data/memory.json

The memory schema has:
- profile: stable user identity, role, skills, preferences
- facts: explicit long-term facts added from the UI or API
- conversations: recent chat turns automatically stored after responses

Browser UI:
- Use the side panel to add memory facts.
- Recent facts and conversation turns are shown in the memory space.
- Uploaded documents also add a memory fact.

API examples after login cookie is established in the browser:

  curl http://localhost:8000/memory

  curl -X POST http://localhost:8000/memory/facts \
    -H "Content-Type: application/json" \
    -d '{"text":"Prefer concise command-first answers", "tags":["preference"]}'

Chat API
--------

The browser uses:

  POST http://localhost:8000/chat

Payload:

  {
    "message": "Explain blockchain data pipelines",
    "history": [],
    "model": "gemma4:12b"
  }

The backend forwards to the private core app at http://app:8000/ask and streams Ollama NDJSON back to the browser.

Security Notes
--------------

- Public ports are bound to 127.0.0.1 only.
- Chat, upload, and memory APIs require a signed HTTP-only session cookie.
- Upload filenames are sanitized and allowed extensions are enforced.
- The frontend no longer injects user messages as HTML.
- Markdown-like assistant rendering is escaped before formatting.
- CORS is restricted by ALLOWED_ORIGINS.

Operational Notes
-----------------

- Keep .env private. It is ignored by git.
- The default Compose fallback password is changeme; replace it in .env before using the assistant.
- data/docs and data/vectordb are ignored by git because they can contain private documents.
- data/memory.json is intentionally present as the editable default memory space.
