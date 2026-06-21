import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

DEFAULT_MEMORY = {
    "profile": {
        "name": "Suganeshwaran",
        "role": "AI & Blockchain Engineer",
        "skills": ["Python", "Docker", "Web3", "Airflow"],
        "preferences": {
            "response_style": "direct and technical",
            "privacy_level": "high",
        },
    },
    "facts": [],
    "conversations": [],
}


class MemoryStore:
    def __init__(self, path: str):
        self.path = Path(path)
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write(DEFAULT_MEMORY)

    def load(self) -> dict[str, Any]:
        with self._lock:
            try:
                with self.path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                data = DEFAULT_MEMORY.copy()
            return self._normalize(data)

    def save(self, data: dict[str, Any]) -> dict[str, Any]:
        normalized = self._normalize(data)
        with self._lock:
            self._write(normalized)
        return normalized

    def update_profile(self, profile_patch: dict[str, Any]) -> dict[str, Any]:
        data = self.load()
        profile = data.setdefault("profile", {})
        self._deep_update(profile, profile_patch)
        data["updated_at"] = self._now()
        return self.save(data)

    def add_fact(self, text: str, tags: list[str] | None = None) -> dict[str, Any]:
        data = self.load()
        fact = {
            "id": uuid4().hex,
            "text": text.strip(),
            "tags": tags or [],
            "created_at": self._now(),
        }
        data.setdefault("facts", []).append(fact)
        data["updated_at"] = self._now()
        self.save(data)
        return fact

    def add_conversation(self, user: str, assistant: str, model: str) -> dict[str, Any]:
        data = self.load()
        turn = {
            "id": uuid4().hex,
            "user": user.strip(),
            "assistant": assistant.strip(),
            "model": model,
            "created_at": self._now(),
        }
        conversations = data.setdefault("conversations", [])
        conversations.append(turn)
        max_turns = int(os.getenv("MAX_MEMORY_TURNS", "200"))
        if len(conversations) > max_turns:
            data["conversations"] = conversations[-max_turns:]
        data["updated_at"] = self._now()
        self.save(data)
        return turn

    def prompt_context(self, recent_turns: int = 5) -> str:
        data = self.load()
        profile = data.get("profile", {})
        skills = profile.get("skills") or []
        if isinstance(skills, str):
            skills = [skills]
        preferences = profile.get("preferences") or {}
        facts = data.get("facts", [])[-20:]
        conversations = data.get("conversations", [])[-recent_turns:]

        lines = [
            "User Profile:",
            f"Name: {profile.get('name', '')}",
            f"Role: {profile.get('role', '')}",
            f"Skills: {', '.join(skills)}",
        ]
        if preferences:
            lines.append("Preferences:")
            for key, value in preferences.items():
                lines.append(f"- {key}: {value}")
        if facts:
            lines.append("Stored Memory Facts:")
            for fact in facts:
                lines.append(f"- {fact.get('text', '')}")
        if conversations:
            lines.append("Recent Stored Conversation Memory:")
            for turn in conversations:
                lines.append(f"- User: {turn.get('user', '')}")
                lines.append(f"  Assistant: {turn.get('assistant', '')[:500]}")
        return "\n".join(lines)

    def _write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=self.path.parent, prefix="memory-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.write("\n")
            os.replace(tmp_name, self.path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)

    def _normalize(self, data: dict[str, Any]) -> dict[str, Any]:
        if "profile" not in data:
            profile = {
                "name": data.get("name", DEFAULT_MEMORY["profile"]["name"]),
                "role": data.get("role", DEFAULT_MEMORY["profile"]["role"]),
                "skills": data.get("skills", DEFAULT_MEMORY["profile"]["skills"]),
                "preferences": data.get("preferences", DEFAULT_MEMORY["profile"]["preferences"]),
            }
            data = {"profile": profile, "facts": [], "conversations": []}
        data.setdefault("profile", {})
        data.setdefault("facts", [])
        data.setdefault("conversations", [])
        data.setdefault("updated_at", self._now())
        return data

    def _deep_update(self, target: dict[str, Any], patch: dict[str, Any]) -> None:
        for key, value in patch.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()
