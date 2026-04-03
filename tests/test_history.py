"""Tests for LessonHistory."""


import pytest

from microtutor.history import LessonHistory, LessonRecord


@pytest.fixture
def history(tmp_path):
    return LessonHistory(tmp_path / "lessons.jsonl")


def _make_record(concept_id: str = "vectors", **kwargs) -> LessonRecord:
    defaults = dict(
        concept_id=concept_id,
        concept_name="Vectors",
        started_at="2026-04-01T12:00:00Z",
        ended_at="2026-04-01T12:08:00Z",
        duration_seconds=480,
        mastery_before=0.05,
        mastery_after=0.85,
        key_topics_covered=["what is a vector", "addition"],
        examples_used=["RGB colors", "walking directions"],
        summary="Taught vectors using RGB analogy.",
        conversation_digest="Explained vectors, student grasped addition.",
    )
    defaults.update(kwargs)
    return LessonRecord(**defaults)


def test_append_and_load(history):
    record = _make_record()
    history.append(record)

    loaded = history.load_all()
    assert len(loaded) == 1
    assert loaded[0].concept_id == "vectors"
    assert loaded[0].mastery_after == 0.85


def test_load_recent(history):
    for i in range(10):
        history.append(_make_record(concept_id=f"concept_{i}"))

    recent = history.load_recent(3)
    assert len(recent) == 3
    assert recent[0].concept_id == "concept_7"
    assert recent[2].concept_id == "concept_9"


def test_get_last_lesson(history):
    assert history.get_last_lesson() is None

    history.append(_make_record(concept_id="a"))
    history.append(_make_record(concept_id="b"))

    last = history.get_last_lesson()
    assert last.concept_id == "b"


def test_build_context_injection_empty(history):
    result = history.build_context_injection()
    assert result == ""


def test_build_context_injection_with_records(history):
    history.append(_make_record(concept_id="vectors", concept_name="Vectors"))
    history.append(_make_record(concept_id="dot_product", concept_name="Dot Product"))

    result = history.build_context_injection()
    assert "PREVIOUS LESSONS" in result
    assert "Vectors" in result
    assert "Dot Product" in result
    assert "Do not repeat examples" in result


def test_build_context_injection_summary_vs_full(history):
    # Add 6 records: 3 should get summaries, 3 should get full digests
    for i in range(6):
        history.append(_make_record(
            concept_id=f"c{i}",
            concept_name=f"Concept {i}",
            summary=f"Summary for concept {i}.",
            conversation_digest=f"Digest for concept {i}.",
        ))

    result = history.build_context_injection(n_full=3, n_summary=10)
    # Should contain both summaries and full digests
    assert "PREVIOUS LESSONS" in result
    assert "Concept 5" in result  # most recent, full digest


def test_load_empty_file(history):
    history.lessons_path.touch()
    assert history.load_all() == []


def test_load_with_corrupt_line(history):
    with open(history.lessons_path, "w") as f:
        f.write("not json\n")
        f.write('{"concept_id": "a", "concept_name": "A", "started_at": "", "ended_at": "", "duration_seconds": 0, "mastery_before": 0, "mastery_after": 0}\n')

    records = history.load_all()
    assert len(records) == 1
    assert records[0].concept_id == "a"
