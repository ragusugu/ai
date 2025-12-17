import os
import json
from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
import chromadb

DATA_DIR = "/data/docs"
DB_DIR = "/data/vectordb"

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.Client(
    chromadb.config.Settings(persist_directory=DB_DIR)
)
collection = client.get_or_create_collection("personal_knowledge")

def read_file(path):
    if path.endswith(".pdf"):
        reader = PdfReader(path)
        return "\n".join(p.page_content for p in reader.pages)
    elif path.endswith(".docx"):
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    elif path.endswith(".json"):
        with open(path) as f:
            return json.dumps(json.load(f), indent=2)
    elif path.endswith(".txt"):
        with open(path) as f:
            return f.read()
    return ""

def chunk(text, size=500):
    words = text.split()
    for i in range(0, len(words), size):
        yield " ".join(words[i:i+size])

for file in os.listdir(DATA_DIR):
    path = os.path.join(DATA_DIR, file)
    # text = read_file(path)
    if not os.path.isfile(path):
        continue

    text = read_file(path)
    if not text.strip():
        continue

    for i, ch in enumerate(chunk(text)):
        embedding = model.encode(ch).tolist()
        collection.add(
            documents=[ch],
            embeddings=[embedding],
            ids=[f"{file}_{i}"]
        )

client.persist()
print("✅ Documents indexed successfully")
