from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import base64
import hashlib
import hmac
import json
import os
import requests
import secrets
import time

app = FastAPI(title="Private AI Assistant UI")

APP_BASE_URL = os.getenv("APP_BASE_URL", "http://app:8000")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemma4:12b")
AUTH_USERNAME = os.getenv("AI_USERNAME", "sugan")
AUTH_PASSWORD = os.getenv("AI_PASSWORD", "changeme")
SESSION_SECRET = os.getenv("SESSION_SECRET", "change-this-local-secret")
SESSION_COOKIE = "pai_session"
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "86400"))


class Prompt(BaseModel):
    message: str = Field(..., min_length=1, max_length=20000)
    model: Optional[str] = DEFAULT_MODEL
    history: Optional[List[Dict[str, str]]] = []


class LoginPayload(BaseModel):
    username: str
    password: str


class MemoryFactPayload(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    tags: List[str] = []


class ProfilePatchPayload(BaseModel):
    profile: Dict[str, Any]


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _unb64(data: str) -> bytes:
    padded = data + "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))


def _sign(payload: str) -> str:
    return hmac.new(SESSION_SECRET.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def create_session(username: str) -> str:
    payload = _b64(json.dumps({"sub": username, "iat": int(time.time())}, separators=(",", ":")).encode("utf-8"))
    return f"{payload}.{_sign(payload)}"


def verify_session(token: str | None) -> bool:
    if not token or "." not in token:
        return False
    payload, signature = token.rsplit(".", 1)
    if not hmac.compare_digest(signature, _sign(payload)):
        return False
    try:
        data = json.loads(_unb64(payload).decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return False
    return data.get("sub") == AUTH_USERNAME and int(time.time()) - int(data.get("iat", 0)) <= SESSION_TTL_SECONDS


def require_auth(request: Request) -> None:
    if not verify_session(request.cookies.get(SESSION_COOKIE)):
        raise HTTPException(status_code=401, detail="Authentication required")


def _proxy_json(method: str, path: str, **kwargs):
    try:
        r = requests.request(method, f"{APP_BASE_URL}{path}", timeout=120, **kwargs)
        if r.headers.get("content-type", "").startswith("application/json"):
            payload = r.json()
        else:
            payload = {"detail": r.text}
        if r.status_code >= 400:
            raise HTTPException(status_code=r.status_code, detail=payload.get("detail", payload))
        return payload
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=str(e)) from e


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/session")
def session(request: Request) -> dict[str, Any]:
    return {"authenticated": verify_session(request.cookies.get(SESSION_COOKIE)), "username": AUTH_USERNAME}


@app.post("/login")
def login(payload: LoginPayload, response: Response) -> dict[str, str]:
    valid_user = secrets.compare_digest(payload.username, AUTH_USERNAME)
    valid_password = secrets.compare_digest(payload.password, AUTH_PASSWORD)
    if not (valid_user and valid_password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    response.set_cookie(
        SESSION_COOKIE,
        create_session(payload.username),
        httponly=True,
        samesite="lax",
        max_age=SESSION_TTL_SECONDS,
    )
    return {"status": "ok"}


@app.post("/logout")
def logout(response: Response) -> dict[str, str]:
    response.delete_cookie(SESSION_COOKIE)
    return {"status": "ok"}


@app.post("/chat")
def chat(prompt: Prompt, _: None = Depends(require_auth)):
    payload = {"prompt": prompt.message, "model": prompt.model or DEFAULT_MODEL, "history": prompt.history}
    try:
        r = requests.post(f"{APP_BASE_URL}/ask", json=payload, stream=True, timeout=120)
        r.raise_for_status()

        def generate():
            for chunk in r.iter_content(chunk_size=None):
                if chunk:
                    yield chunk

        return StreamingResponse(generate(), media_type="application/x-ndjson")
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=str(e)) from e


@app.post("/upload")
async def upload(file: UploadFile = File(...), _: None = Depends(require_auth)):
    files = {"file": (file.filename, file.file, file.content_type)}
    return _proxy_json("POST", "/upload", files=files)


@app.get("/memory")
def get_memory(_: None = Depends(require_auth)):
    return _proxy_json("GET", "/memory")


@app.patch("/memory/profile")
def update_profile(payload: ProfilePatchPayload, _: None = Depends(require_auth)):
    return _proxy_json("PATCH", "/memory/profile", json=payload.model_dump())


@app.post("/memory/facts")
def add_memory_fact(payload: MemoryFactPayload, _: None = Depends(require_auth)):
    return _proxy_json("POST", "/memory/facts", json=payload.model_dump())


allowed_origins = [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST", "PATCH"],
    allow_headers=["Content-Type"],
    allow_credentials=True,
)


@app.exception_handler(HTTPException)
def http_exception_handler(_: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


app.mount("/", StaticFiles(directory="ui", html=True), name="ui")
