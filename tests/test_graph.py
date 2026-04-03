"""Tests for ConceptGraph."""

import json
import tempfile
from pathlib import Path

import pytest

from microtutor.graph import ConceptGraph


@pytest.fixture
def sample_graph_path():
    data = {
        "title": "Test Graph",
        "concepts": [
            {
                "id": "a",
                "name": "Concept A",
                "description": "Root concept",
                "prerequisites": [],
                "teaching_hints": ["hint1"],
            },
            {
                "id": "b",
                "name": "Concept B",
                "description": "Depends on A",
                "prerequisites": ["a"],
                "teaching_hints": [],
            },
            {
                "id": "c",
                "name": "Concept C",
                "description": "Depends on A and B",
                "prerequisites": ["a", "b"],
            },
        ],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        f.flush()
    return Path(f.name)


def test_load_graph(sample_graph_path):
    graph = ConceptGraph.from_json(sample_graph_path)
    assert len(graph.get_all_concept_ids()) == 3


def test_topological_order(sample_graph_path):
    graph = ConceptGraph.from_json(sample_graph_path)
    order = graph.get_all_concept_ids()
    assert order.index("a") < order.index("b")
    assert order.index("b") < order.index("c")


def test_get_prerequisites(sample_graph_path):
    graph = ConceptGraph.from_json(sample_graph_path)
    assert graph.get_prerequisites("a") == []
    assert graph.get_prerequisites("b") == ["a"]
    assert set(graph.get_prerequisites("c")) == {"a", "b"}


def test_get_node(sample_graph_path):
    graph = ConceptGraph.from_json(sample_graph_path)
    node = graph.get_node("a")
    assert node.name == "Concept A"
    assert node.teaching_hints == ["hint1"]


def test_get_node_unknown():
    data = {"title": "Empty", "concepts": []}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = Path(f.name)
    graph = ConceptGraph.from_json(path)
    with pytest.raises(KeyError):
        graph.get_node("nonexistent")


def test_topological_depth(sample_graph_path):
    graph = ConceptGraph.from_json(sample_graph_path)
    assert graph.get_topological_depth("a") == 0
    assert graph.get_topological_depth("b") == 1
    assert graph.get_topological_depth("c") == 2


def test_cycle_detection():
    data = {
        "title": "Cyclic",
        "concepts": [
            {"id": "x", "name": "X", "description": "x", "prerequisites": ["y"]},
            {"id": "y", "name": "Y", "description": "y", "prerequisites": ["x"]},
        ],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = Path(f.name)
    with pytest.raises(ValueError, match="cycles"):
        ConceptGraph.from_json(path)


def test_missing_prerequisite():
    data = {
        "title": "Bad ref",
        "concepts": [
            {"id": "a", "name": "A", "description": "a", "prerequisites": ["missing"]},
        ],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = Path(f.name)
    with pytest.raises(ValueError, match="not found"):
        ConceptGraph.from_json(path)


def test_real_neural_networks_graph():
    """Smoke test against the actual data file."""
    path = Path(__file__).parent.parent / "data" / "neural_networks.json"
    graph = ConceptGraph.from_json(path)
    assert len(graph.get_all_concept_ids()) == 15
    # Vectors should be at depth 0 (no prerequisites)
    assert graph.get_topological_depth("vectors") == 0
    # MLP should be the deepest
    assert graph.get_topological_depth("mlp") > 0
