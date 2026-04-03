"""Tests for StudentModel and BKT update logic."""

import json
import tempfile
from pathlib import Path

import pytest

from microtutor.graph import ConceptGraph
from microtutor.model import StudentModel


@pytest.fixture
def simple_graph():
    data = {
        "title": "Test",
        "concepts": [
            {"id": "a", "name": "A", "description": "a", "prerequisites": []},
            {"id": "b", "name": "B", "description": "b", "prerequisites": ["a"]},
        ],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        f.flush()
    return ConceptGraph.from_json(Path(f.name))


def test_initial_mastery(simple_graph):
    model = StudentModel(simple_graph)
    # Default p_init is 0.05
    assert model.predict_mastery("a") == pytest.approx(0.05)
    assert model.predict_mastery("b") == pytest.approx(0.05)


def test_correct_answer_increases_mastery(simple_graph):
    model = StudentModel(simple_graph)
    before = model.predict_mastery("a")
    model.update("a", correct=True)
    after = model.predict_mastery("a")
    assert after > before


def test_incorrect_answer_lower_increase(simple_graph):
    # Two models: one gets correct, one gets incorrect
    model_correct = StudentModel(simple_graph)
    model_incorrect = StudentModel(simple_graph)
    model_correct.update("a", correct=True)
    model_incorrect.update("a", correct=False)
    assert model_correct.predict_mastery("a") > model_incorrect.predict_mastery("a")


def test_mastery_convergence(simple_graph):
    """Many correct answers should push mastery close to 1.0."""
    model = StudentModel(simple_graph)
    for _ in range(20):
        model.update("a", correct=True)
    assert model.predict_mastery("a") > 0.95


def test_mastery_stays_bounded(simple_graph):
    """Mastery should stay in [0, 1]."""
    model = StudentModel(simple_graph)
    for _ in range(50):
        model.update("a", correct=True)
    assert 0.0 <= model.predict_mastery("a") <= 1.0
    for _ in range(50):
        model.update("a", correct=False)
    assert 0.0 <= model.predict_mastery("a") <= 1.0


def test_attempt_count(simple_graph):
    model = StudentModel(simple_graph)
    assert model.get_attempt_count("a") == 0
    model.update("a", correct=True)
    assert model.get_attempt_count("a") == 1
    model.update("a", correct=False)
    assert model.get_attempt_count("a") == 2


def test_save_and_load_round_trip(simple_graph):
    model = StudentModel(simple_graph)
    model.update("a", correct=True)
    model.update("a", correct=True)
    model.update("b", correct=False)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)

    model.save(path)

    model2 = StudentModel(simple_graph)
    model2.load(path)

    assert model2.predict_mastery("a") == pytest.approx(model.predict_mastery("a"))
    assert model2.predict_mastery("b") == pytest.approx(model.predict_mastery("b"))
    assert model2.get_attempt_count("a") == 2
    assert model2.get_attempt_count("b") == 1


def test_load_nonexistent_file(simple_graph):
    """Loading a nonexistent file should keep initial state."""
    model = StudentModel(simple_graph)
    model.load(Path("/tmp/nonexistent_student_12345.json"))
    assert model.predict_mastery("a") == pytest.approx(0.05)


def test_save_creates_parent_dirs(simple_graph):
    model = StudentModel(simple_graph)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "subdir" / "student.json"
        model.save(path)
        assert path.exists()


def test_schema_version_in_save(simple_graph):
    model = StudentModel(simple_graph)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)
    model.save(path)
    with open(path) as f:
        data = json.load(f)
    assert data["schema_version"] == 2
    assert "last_updated_at" in data["states"]["a"]


def test_observation_log(simple_graph):
    model = StudentModel(simple_graph)
    model.update("a", correct=True)
    model.update("b", correct=False)
    assert len(model._observation_log) == 2
    assert model._observation_log[0]["concept_id"] == "a"
    assert model._observation_log[0]["correct"] is True
    assert model._observation_log[1]["concept_id"] == "b"
    assert model._observation_log[1]["correct"] is False
