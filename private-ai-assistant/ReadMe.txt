private-ai-assistant/
│
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── app/
│   ├── main.py
│   └── system_prompt.txt
└── data/
    └── memory.json


🚀 Run cmds:

  > docker compose up -d
  > docker compose down
  > docker compose build

You changed docker-compose.yml or Dockerfile ⚠️
Any change to:

  Dockerfile
  docker-compose.yml
  requirements.txt

Always requires rebuild

  > docker compose down
  > docker compose up -d --build

Pull model (once):

  > docker exec -it ollama ollama pull llama3 
  or 
  > docker exec -it ollama ollama pull llama3:8b-instruct-q4_K_M


Test Your Assistant

  > curl -X POST http://localhost:8000/ask \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Explain blockchain data pipelines"}'
