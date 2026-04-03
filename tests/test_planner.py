"""Tests for the Curriculum Planner."""

import json
import tempfile
from pathlib import Path

import pytest

from microtutor.graph import ConceptGraph
from microtutor.model import StudentModel
from microtutor.planner import (
    MAX_ATTEMPTS_BEFORE_BACKTRACK,
    PREREQ_MASTERY_THRESHOLD,
    TEACH_MASTERY_THRESHOLD,
    Planner,
)


def make_graph(concepts):
    data = {"title": "Test", "concepts": concepts}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        f.flush()
    return ConceptGraph.from_json(Path(f.name))


@pytest.fixture
def linear_graph():
    """a -> b -> c (linear chain)."""
    return make_graph([
        {"id": "a", "name": "A", "description": "a", "prerequisites": []},
        {"id": "b", "name": "B", "description": "b", "prerequisites": ["a"]},
        {"id": "c", "name": "C", "description": "c", "prerequisites": ["b"]},
    ])


@pytest.fixture
def diamond_graph():
    """a -> b, a -> c, b+c -> d (diamond)."""
    return make_graph([
        {"id": "a", "name": "A", "description": "a", "prerequisites": []},
        {"id": "b", "name": "B", "description": "b", "prerequisites": ["a"]},
        {"id": "c", "name": "C", "description": "c", "prerequisites": ["a"]},
        {"id": "d", "name": "D", "description": "d", "prerequisites": ["b", "c"]},
    ])


def test_initial_frontier_is_roots(linear_graph):
    model = StudentModel(linear_graph)
    planner = Planner(linear_graph, model)
    frontier = planner.get_frontier()
    assert frontier == ["a"]


def test_frontier_advances_after_mastery(linear_graph):
    model = StudentModel(linear_graph)
    planner = Planner(linear_graph, model)
    # Master concept 'a'
    for _ in range(15):
        model.update("a", correct=True)
    assert model.predict_mastery("a") > PREREQ_MASTERY_THRESHOLD
    frontier = planner.get_frontier()
    assert "b" in frontier
    assert "a" not in frontier  # mastered, no longer below TEACH threshold


def test_frontier_respects_prerequisites(diamond_graph):
    model = StudentModel(diamond_graph)
    planner = Planner(diamond_graph, model)
    # Only 'a' should be on frontier initially
    frontier = planner.get_frontier()
    assert frontier == ["a"]
    # d should never be on frontier until b and c are mastered
    assert "d" not in frontier


def test_select_next_concept_shallowest_first(diamond_graph):
    model = StudentModel(diamond_graph)
    planner = Planner(diamond_graph, model)
    # Master 'a' so b and c are on frontier
    for _ in range(15):
        model.update("a", correct=True)

    concept = planner.select_next_concept()
    # Both b and c are at depth 1, either is valid
    assert concept in ("b", "c")


def test_completion_detection(linear_graph):
    model = StudentModel(linear_graph)
    planner = Planner(linear_graph, model)
    assert not planner.is_complete()

    # Master all concepts
    for cid in ["a", "b", "c"]:
        for _ in range(15):
            model.update(cid, correct=True)

    assert planner.is_complete()


def test_select_returns_none_when_complete(linear_graph):
    model = StudentModel(linear_graph)
    planner = Planner(linear_graph, model)
    for cid in ["a", "b", "c"]:
        for _ in range(15):
            model.update(cid, correct=True)
    assert planner.select_next_concept() is None


def test_backtrack_after_repeated_failure(linear_graph):
    model = StudentModel(linear_graph)
    planner = Planner(linear_graph, model)

    # Master 'a' enough to unlock 'b'
    for _ in range(15):
        model.update("a", correct=True)

    # Now manually lower a's mastery to simulate overestimation
    model.states["a"].mastery = 0.5

    # Fail 'b' multiple times
    for _ in range(MAX_ATTEMPTS_BEFORE_BACKTRACK):
        model.update("b", correct=False)

    # Planner should backtrack to 'a' (weakest prereq below threshold)
    next_concept = planner.select_next_concept()
    assert next_concept == "a"


def test_stuck_detection(linear_graph):
    model = StudentModel(linear_graph)
    planner = Planner(linear_graph, model)

    # Set 'a' mastery to between thresholds (above TEACH but below PREREQ)
    # so 'a' is not on frontier (mastery >= TEACH) but 'b' is blocked (a < PREREQ)
    model.states["a"].mastery = 0.5  # above 0.4 but below 0.7

    assert planner.is_stuck()


def test_build_context(linear_graph):
    model = StudentModel(linear_graph)
    planner = Planner(linear_graph, model)
    context = planner.build_context("a")
    assert context.concept_id == "a"
    assert context.concept_name == "A"
    assert context.prerequisite_mastery == {}
    assert context.attempt_number == 1
