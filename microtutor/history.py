"""Lesson history recording and retrieval for microtutor."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class LessonRecord:
    concept_id: str
    concept_name: str
    started_at: str
    ended_at: str
    duration_seconds: float
    mastery_before: float
    mastery_after: float
    key_topics_covered: list[str] = field(default_factory=list)
    examples_used: list[str] = field(default_factory=list)
    assessment_results: list[dict] = field(default_factory=list)
    summary: str = ""
    conversation_digest: str = ""


class LessonHistory:
    """Manages the lessons.jsonl file for a course."""

    def __init__(self, lessons_path: Path) -> None:
        self.lessons_path = lessons_path

    def append(self, record: LessonRecord) -> None:
        """Append a lesson record to the JSONL file."""
        self.lessons_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.lessons_path, "a") as f:
            f.write(json.dumps(asdict(record)) + "\n")

    def load_all(self) -> list[LessonRecord]:
        """Load all lesson records."""
        if not self.lessons_path.exists():
            return []
        records = []
        with open(self.lessons_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    records.append(LessonRecord(**{
                        k: v for k, v in data.items()
                        if k in LessonRecord.__dataclass_fields__
                    }))
                except (json.JSONDecodeError, TypeError):
                    continue
        return records

    def load_recent(self, n: int = 5) -> list[LessonRecord]:
        """Load the most recent n lesson records."""
        all_records = self.load_all()
        return all_records[-n:]

    def build_context_injection(
        self, n_full: int = 3, n_summary: int = 10
    ) -> str:
        """Build a text block for injection into the teaching system prompt.

        Last n_full lessons get full digests. The next n_summary get one-line
        summaries. Older lessons are omitted.
        """
        all_records = self.load_all()
        if not all_records:
            return ""

        lines = ["PREVIOUS LESSONS IN THIS COURSE:"]

        # Recent lessons with full digest
        recent = all_records[-(n_full + n_summary):]
        full_start = max(0, len(recent) - n_full)

        # One-line summaries for older records in the window
        for record in recent[:full_start]:
            line = (
                f"- {record.concept_name}: {record.summary}"
                if record.summary
                else f"- {record.concept_name}: mastery {record.mastery_before:.0%} -> {record.mastery_after:.0%}"
            )
            lines.append(line)

        # Full digests for most recent
        for record in recent[full_start:]:
            examples = ", ".join(record.examples_used) if record.examples_used else "none noted"
            digest = record.conversation_digest or record.summary or "no digest"
            ago = _format_lesson_time(record)
            lines.append(
                f"- Lesson: {record.concept_name} ({ago}): {digest} "
                f"Examples used: {examples}. "
                f"Mastery: {record.mastery_before:.0%} -> {record.mastery_after:.0%}."
            )

        lines.append("")
        lines.append(
            "IMPORTANT: Do not repeat examples from previous lessons. "
            "Reference them: 'recall when we used [example] for [concept]' "
            "rather than re-explaining from scratch. "
            "The student has written notes from each lesson they can review offline."
        )

        return "\n".join(lines)

    def get_last_lesson(self) -> LessonRecord | None:
        """Get the most recent lesson record, or None."""
        all_records = self.load_all()
        return all_records[-1] if all_records else None


def _format_lesson_time(record: LessonRecord) -> str:
    """Format lesson timing for display."""
    mins = int(record.duration_seconds // 60)
    if mins > 0:
        return f"{mins} min"
    return f"{int(record.duration_seconds)}s"
