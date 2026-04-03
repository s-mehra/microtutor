"""Tests for CourseManager."""

import json

import pytest

from microtutor.course import CourseManager


@pytest.fixture
def course_dir(tmp_path):
    """A temporary course directory with a simple graph."""
    course = tmp_path / "test-course"
    course.mkdir()

    graph_data = {
        "title": "Test Course",
        "description": "A test",
        "concepts": [
            {"id": "a", "name": "A", "description": "a", "prerequisites": []},
            {"id": "b", "name": "B", "description": "b", "prerequisites": ["a"]},
        ],
    }
    with open(course / "graph.json", "w") as f:
        json.dump(graph_data, f)

    meta = {
        "title": "Test Course",
        "slug": "test-course",
        "description": "A test",
        "created_at": "2026-04-01T00:00:00Z",
        "last_session_at": "",
        "total_lessons_completed": 0,
        "total_concepts": 2,
        "concepts_mastered": 0,
        "schema_version": 1,
    }
    with open(course / "meta.json", "w") as f:
        json.dump(meta, f)

    (course / "lessons.jsonl").touch()

    return course


def test_load_meta(course_dir):
    cm = CourseManager(course_dir)
    meta = cm.load_meta()
    assert meta.title == "Test Course"
    assert meta.total_concepts == 2


def test_save_meta(course_dir):
    cm = CourseManager(course_dir)
    meta = cm.load_meta()
    meta.total_lessons_completed = 5
    cm.save_meta(meta)

    reloaded = cm.load_meta()
    assert reloaded.total_lessons_completed == 5


def test_load_graph(course_dir):
    cm = CourseManager(course_dir)
    graph = cm.load_graph()
    assert len(graph.get_all_concept_ids()) == 2


def test_load_and_save_student(course_dir):
    cm = CourseManager(course_dir)
    graph = cm.load_graph()
    model = cm.load_student(graph)

    # Fresh student
    assert model.predict_mastery("a") == pytest.approx(0.05)

    # Update and save
    model.update("a", correct=True)
    cm.save_student(model)

    # Reload
    model2 = cm.load_student(graph)
    assert model2.predict_mastery("a") > 0.05


def test_has_lessons_empty(course_dir):
    cm = CourseManager(course_dir)
    assert cm.has_lessons() is False


def test_has_lessons_with_data(course_dir):
    cm = CourseManager(course_dir)
    with open(course_dir / "lessons.jsonl", "w") as f:
        f.write('{"concept_id": "a"}\n')
    assert cm.has_lessons() is True


def test_update_meta_stats(course_dir):
    cm = CourseManager(course_dir)
    graph = cm.load_graph()
    model = cm.load_student(graph)

    # Master concept 'a'
    for _ in range(15):
        model.update("a", correct=True)

    meta = cm.load_meta()
    cm.update_meta_stats(meta, model, graph)

    reloaded = cm.load_meta()
    assert reloaded.concepts_mastered == 1
    assert reloaded.last_session_at != ""
