"""Tests for ConfigManager."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from microtutor.config import ConfigManager, AppConfig, _slugify


@pytest.fixture
def config_manager(tmp_path):
    """ConfigManager with a temporary directory."""
    cm = ConfigManager()
    cm.CONFIG_DIR = tmp_path
    cm.CONFIG_PATH = tmp_path / "config.json"
    cm.COURSES_DIR = tmp_path / "courses"
    return cm


def test_exists_false(config_manager):
    assert config_manager.exists() is False


def test_save_and_load(config_manager):
    app_config = AppConfig(name="Alice", api_key="sk-test-123", created_at="2026-01-01T00:00:00Z")
    config_manager.save(app_config)
    assert config_manager.exists()

    loaded = config_manager.load()
    assert loaded.name == "Alice"
    assert loaded.api_key == "sk-test-123"


def test_get_api_key_from_config(config_manager):
    app_config = AppConfig(name="Alice", api_key="sk-saved")
    config_manager.save(app_config)
    assert config_manager.get_api_key() == "sk-saved"


def test_get_api_key_env_override(config_manager):
    app_config = AppConfig(name="Alice", api_key="sk-saved")
    config_manager.save(app_config)
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-env"}):
        assert config_manager.get_api_key() == "sk-env"


def test_list_courses_empty(config_manager):
    config_manager.COURSES_DIR.mkdir(parents=True, exist_ok=True)
    assert config_manager.list_courses() == []


def test_list_courses(config_manager):
    courses_dir = config_manager.COURSES_DIR
    courses_dir.mkdir(parents=True)

    course_dir = courses_dir / "test-course-abc1234"
    course_dir.mkdir()
    meta = {
        "title": "Test Course",
        "slug": "test-course-abc1234",
        "last_session_at": "2026-04-01T12:00:00Z",
        "total_concepts": 10,
        "concepts_mastered": 3,
    }
    with open(course_dir / "meta.json", "w") as f:
        json.dump(meta, f)

    courses = config_manager.list_courses()
    assert len(courses) == 1
    assert courses[0].title == "Test Course"
    assert courses[0].total_concepts == 10
    assert courses[0].concepts_mastered == 3
    assert courses[0].progress_pct == pytest.approx(0.3)


def test_create_course_dir(config_manager):
    config_manager.COURSES_DIR.mkdir(parents=True)
    course_dir = config_manager.create_course_dir("Intro to Neural Networks")
    assert course_dir.exists()
    assert "intro-to-neural-networks" in course_dir.name


def test_slugify():
    assert _slugify("Hello World!") == "hello-world"
    assert _slugify("Intro to Neural Networks") == "intro-to-neural-networks"
    assert _slugify("  lots   of   spaces  ") == "lots-of-spaces"
    assert _slugify("special@#$chars") == "specialchars"
