"""Tests for knowledge decay."""

import json
import tempfile
import time
from pathlib import Path

import pytest

from microtutor.decay import apply_decay, apply_decay_to_model
from microtutor.graph import ConceptGraph
from microtutor.model import StudentModel


def test_no_decay_when_just_updated():
    now = time.time()
    result = apply_decay(0.85, last_updated_at=now, current_time=now)
    assert result == pytest.approx(0.85)


def test_half_life_decay():
    now = time.time()
    fourteen_days_ago = now - (14 * 86400)
    result = apply_decay(0.85, last_updated_at=fourteen_days_ago, current_time=now)
    # After one half-life: floor + (0.85 - floor) * 0.5
    expected = 0.05 + (0.85 - 0.05) * 0.5
    assert result == pytest.approx(expected)


def test_double_half_life():
    now = time.time()
    twenty_eight_days_ago = now - (28 * 86400)
    result = apply_decay(0.85, last_updated_at=twenty_eight_days_ago, current_time=now)
    expected = 0.05 + (0.85 - 0.05) * 0.25
    assert result == pytest.approx(expected)


def test_never_below_floor():
    now = time.time()
    very_long_ago = now - (365 * 86400)
    result = apply_decay(0.85, last_updated_at=very_long_ago, current_time=now)
    assert result >= 0.05


def test_no_decay_if_already_at_floor():
    now = time.time()
    long_ago = now - (14 * 86400)
    result = apply_decay(0.05, last_updated_at=long_ago, current_time=now)
    assert result == pytest.approx(0.05)


def test_no_decay_if_no_timestamp():
    result = apply_decay(0.85, last_updated_at=0, current_time=time.time())
    assert result == pytest.approx(0.85)


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


def test_apply_decay_to_model(simple_graph):
    model = StudentModel(simple_graph)

    # Set mastery and timestamp to 14 days ago
    now = time.time()
    fourteen_days_ago = now - (14 * 86400)
    model.states["a"].mastery = 0.85
    model.states["a"].last_updated_at = fourteen_days_ago
    model.states["b"].mastery = 0.05  # at floor, should not decay

    decayed = apply_decay_to_model(model, current_time=now)

    # 'a' should have decayed significantly
    assert "a" in decayed
    assert model.predict_mastery("a") < 0.85

    # 'b' should not have decayed (at floor)
    assert "b" not in decayed
    assert model.predict_mastery("b") == pytest.approx(0.05)


def test_small_decay_not_reported(simple_graph):
    model = StudentModel(simple_graph)
    now = time.time()
    one_day_ago = now - 86400
    model.states["a"].mastery = 0.10
    model.states["a"].last_updated_at = one_day_ago

    decayed = apply_decay_to_model(model, current_time=now)
    # Decay from 0.10 after 1 day with 14-day half-life is tiny
    assert "a" not in decayed  # below reporting threshold
