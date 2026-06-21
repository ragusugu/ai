import os
import json
from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
import uuid

DATA_DIR = "/data/docs"
DB_DIR = "/data/vectordb"

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_or_create_collection("personal_knowledge")

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".json"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def read_file(path):
    try:
        if os.path.getsize(path) > MAX_FILE_SIZE:
            print(f"Skipping {path}: file too large (>10MB)")
            return ""
        ext = os.path.splitext(path)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            print(f"Skipping {path}: unsupported extension {ext}")
            return ""
        if path.endswith(".pdf"):
            reader = PdfReader(path)
            return "\n".join((page.extract_text() or "") for page in reader.pages)
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
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return ""

def get_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_text(text)

def run_ingestion():
    if not os.path.isdir(DATA_DIR):
        print(f"No docs directory: {DATA_DIR}")
        return

    for file in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, file)
        if not os.path.isfile(path):
            continue

        text = read_file(path)
        if not text.strip():
            continue

        for i, ch in enumerate(get_chunks(text)):
            embedding = model.encode(ch).tolist()
            collection.add(
                documents=[ch],
                embeddings=[embedding],
                ids=[f"{file}_{i}_{uuid.uuid4().hex}"]
            )

    if hasattr(client, "persist"):
        client.persist()
    print("Documents indexed successfully")

if __name__ == "__main__":
    run_ingestion()
