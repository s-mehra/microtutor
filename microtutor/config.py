"""Configuration management and onboarding for microtutor."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path


SCHEMA_VERSION = 1


@dataclass
class AppConfig:
    name: str
    api_key: str
    created_at: str = ""
    schema_version: int = SCHEMA_VERSION


@dataclass
class CourseSummary:
    slug: str
    title: str
    last_session_at: str
    total_concepts: int
    concepts_mastered: int

    @property
    def progress_pct(self) -> float:
        if self.total_concepts == 0:
            return 0.0
        return self.concepts_mastered / self.total_concepts


class ConfigManager:
    """Manages ~/.microtutor/ directory, config, and course listing."""

    CONFIG_DIR = Path.home() / ".microtutor"
    CONFIG_PATH = CONFIG_DIR / "config.json"
    COURSES_DIR = CONFIG_DIR / "courses"

    def exists(self) -> bool:
        return self.CONFIG_PATH.exists()

    def load(self) -> AppConfig:
        with open(self.CONFIG_PATH) as f:
            data = json.load(f)
        return AppConfig(
            name=data["name"],
            api_key=data["api_key"],
            created_at=data.get("created_at", ""),
            schema_version=data.get("schema_version", 1),
        )

    def save(self, config: AppConfig) -> None:
        self.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        self.COURSES_DIR.mkdir(parents=True, exist_ok=True)
        with open(self.CONFIG_PATH, "w") as f:
            json.dump(asdict(config), f, indent=2)

    def get_api_key(self) -> str:
        """Return API key, with env var taking precedence over saved config."""
        env_key = os.environ.get("ANTHROPIC_API_KEY")
        if env_key:
            return env_key
        if self.exists():
            config = self.load()
            return config.api_key
        return ""

    def validate_api_key(self, key: str) -> bool:
        """Validate an API key by making a lightweight API call."""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=key)
            client.models.list()
            return True
        except Exception:
            return False

    def list_courses(self) -> list[CourseSummary]:
        """Scan courses/ directory and return summaries sorted by last session."""
        if not self.COURSES_DIR.exists():
            return []

        summaries = []
        for course_dir in self.COURSES_DIR.iterdir():
            if not course_dir.is_dir():
                continue
            meta_path = course_dir / "meta.json"
            if not meta_path.exists():
                continue
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                summaries.append(CourseSummary(
                    slug=meta["slug"],
                    title=meta["title"],
                    last_session_at=meta.get("last_session_at", ""),
                    total_concepts=meta.get("total_concepts", 0),
                    concepts_mastered=meta.get("concepts_mastered", 0),
                ))
            except (json.JSONDecodeError, KeyError):
                continue

        # Most recently used first
        summaries.sort(key=lambda s: s.last_session_at, reverse=True)
        return summaries

    def get_course_dir(self, slug: str) -> Path:
        return self.COURSES_DIR / slug

    def create_course_dir(self, title: str) -> Path:
        """Create a new course directory with a unique slug."""
        slug_base = _slugify(title)
        hash_suffix = hashlib.md5(
            f"{title}{time.time()}".encode()
        ).hexdigest()[:7]
        slug = f"{slug_base}-{hash_suffix}"
        course_dir = self.COURSES_DIR / slug
        course_dir.mkdir(parents=True, exist_ok=True)
        return course_dir


def _slugify(text: str) -> str:
    """Convert text to a filesystem-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text[:50].strip("-")
