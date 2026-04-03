"""Course lifecycle management for microtutor."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from microtutor.graph import ConceptGraph
from microtutor.model import StudentModel

META_SCHEMA_VERSION = 1


@dataclass
class CourseMeta:
    title: str
    slug: str
    description: str = ""
    goals: str = ""
    background: str = ""
    desired_depth: str = "intermediate"
    goal_conversation: list[dict] = field(default_factory=list)
    created_at: str = ""
    last_session_at: str = ""
    total_lessons_completed: int = 0
    total_concepts: int = 0
    concepts_mastered: int = 0
    schema_version: int = META_SCHEMA_VERSION


class CourseManager:
    """Manages a single course directory: meta, graph, student state, lessons."""

    def __init__(self, course_dir: Path) -> None:
        self.course_dir = course_dir
        self.meta_path = course_dir / "meta.json"
        self.graph_path = course_dir / "graph.json"
        self.student_path = course_dir / "student.json"
        self.lessons_path = course_dir / "lessons.jsonl"
        self.notes_dir = course_dir / "notes"

    def load_meta(self) -> CourseMeta:
        with open(self.meta_path) as f:
            data = json.load(f)
        return CourseMeta(
            title=data["title"],
            slug=data["slug"],
            description=data.get("description", ""),
            goals=data.get("goals", ""),
            background=data.get("background", ""),
            desired_depth=data.get("desired_depth", "intermediate"),
            goal_conversation=data.get("goal_conversation", []),
            created_at=data.get("created_at", ""),
            last_session_at=data.get("last_session_at", ""),
            total_lessons_completed=data.get("total_lessons_completed", 0),
            total_concepts=data.get("total_concepts", 0),
            concepts_mastered=data.get("concepts_mastered", 0),
        )

    def save_meta(self, meta: CourseMeta) -> None:
        self.course_dir.mkdir(parents=True, exist_ok=True)
        with open(self.meta_path, "w") as f:
            json.dump(asdict(meta), f, indent=2)

    def load_graph(self) -> ConceptGraph:
        return ConceptGraph.from_json(self.graph_path)

    def load_student(self, graph: ConceptGraph) -> StudentModel:
        """Load student state and apply knowledge decay."""
        model = StudentModel(graph)
        if self.student_path.exists():
            model.load(self.student_path)
            # Apply knowledge decay on load
            try:
                from microtutor.decay import apply_decay_to_model
                apply_decay_to_model(model)
            except ImportError:
                pass  # decay module not yet available
        return model

    def save_student(self, model: StudentModel) -> None:
        model.save(self.student_path)

    def update_meta_stats(self, meta: CourseMeta, model: StudentModel, graph: ConceptGraph) -> None:
        """Update meta.json with current stats."""
        from microtutor.planner import PREREQ_MASTERY_THRESHOLD
        meta.last_session_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        meta.total_concepts = len(graph.get_all_concept_ids())
        meta.concepts_mastered = sum(
            1 for cid in graph.get_all_concept_ids()
            if model.predict_mastery(cid) >= PREREQ_MASTERY_THRESHOLD
        )
        self.save_meta(meta)

    def has_lessons(self) -> bool:
        return self.lessons_path.exists() and self.lessons_path.stat().st_size > 0

    def save_note(
        self,
        concept_id: str,
        concept_name: str,
        lesson_title: str,
        conversation: list[dict],
        summary_data: dict,
        assessment_results: list[dict],
        mastery_before: float,
        mastery_after: float,
        duration_seconds: float,
    ) -> Path:
        """Save a lesson as a readable markdown note. Returns the file path."""
        self.notes_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{concept_id}-{timestamp}.md"
        path = self.notes_dir / filename

        mins = int(duration_seconds // 60)
        secs = int(duration_seconds % 60)

        lines = [
            f"# {concept_name} - {lesson_title}",
            f"Date: {time.strftime('%Y-%m-%d %H:%M')}",
            f"Duration: {mins}m {secs}s",
            f"Mastery: {mastery_before:.0%} -> {mastery_after:.0%}",
            "",
            "## Lesson",
            "",
        ]

        for msg in conversation:
            role = "Tutor" if msg["role"] == "assistant" else "You"
            lines.append(f"**{role}:** {msg['content']}")
            lines.append("")

        if summary_data.get("summary"):
            lines.append("## Key Takeaways")
            lines.append("")
            lines.append(summary_data["summary"])
            lines.append("")

        if summary_data.get("examples_used"):
            lines.append("## Examples Used")
            lines.append("")
            for ex in summary_data["examples_used"]:
                lines.append(f"- {ex}")
            lines.append("")

        if assessment_results:
            lines.append("## Assessment")
            lines.append("")
            for r in assessment_results:
                result_str = "Correct" if r.get("correct") else "Incorrect"
                lines.append(f"- {result_str}: {r.get('explanation', '')}")
            lines.append("")

        path.write_text("\n".join(lines))
        return path
