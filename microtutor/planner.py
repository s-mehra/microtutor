"""Curriculum planner that decides what to teach next based on the knowledge graph and student mastery."""

from __future__ import annotations

from dataclasses import dataclass

from microtutor.graph import ConceptGraph
from microtutor.model import StudentModel

PREREQ_MASTERY_THRESHOLD = 0.7
TEACH_MASTERY_THRESHOLD = 0.4
MAX_ATTEMPTS_BEFORE_BACKTRACK = 3
MAX_BACKTRACK_DEPTH = 2


@dataclass
class TeachingContext:
    concept_id: str
    concept_name: str
    description: str
    teaching_hints: list[str]
    prerequisite_mastery: dict[str, float]  # prereq_id -> mastery
    student_mastery: float
    attempt_number: int
    is_backtrack: bool = False
    curriculum_overview: str = ""  # full syllabus with mastery status


class Planner:
    """Selects the next concept to teach based on frontier detection and backtracking."""

    def __init__(self, graph: ConceptGraph, model: StudentModel) -> None:
        self.graph = graph
        self.model = model

    def get_frontier(self) -> list[str]:
        """Find concepts where all prerequisites are met but mastery is low.

        A concept is on the frontier when:
        - All prerequisites have mastery > PREREQ_MASTERY_THRESHOLD
        - The concept itself has mastery < TEACH_MASTERY_THRESHOLD
        """
        frontier = []
        for concept_id in self.graph.get_all_concept_ids():
            mastery = self.model.predict_mastery(concept_id)
            if mastery >= TEACH_MASTERY_THRESHOLD:
                continue

            prereqs = self.graph.get_prerequisites(concept_id)
            if all(
                self.model.predict_mastery(p) >= PREREQ_MASTERY_THRESHOLD
                for p in prereqs
            ):
                frontier.append(concept_id)

        return frontier

    def _find_backtrack_target(self, concept_id: str, depth: int = 0) -> str | None:
        """Find the weakest prerequisite to backtrack to, up to MAX_BACKTRACK_DEPTH."""
        if depth >= MAX_BACKTRACK_DEPTH:
            return None

        prereqs = self.graph.get_prerequisites(concept_id)
        if not prereqs:
            return None

        # Find the prerequisite with the lowest mastery
        weakest = min(prereqs, key=lambda p: self.model.predict_mastery(p))
        weakest_mastery = self.model.predict_mastery(weakest)

        # If the weakest prereq has high mastery, the problem isn't prerequisites
        if weakest_mastery >= PREREQ_MASTERY_THRESHOLD:
            return None

        return weakest

    def select_next_concept(self) -> str | None:
        """Select the next concept to teach.

        Returns None if all concepts are mastered or the student is stuck.
        """
        # Check for backtrack: any concept attempted 3+ times without progress
        for concept_id in self.graph.get_all_concept_ids():
            attempts = self.model.get_attempt_count(concept_id)
            mastery = self.model.predict_mastery(concept_id)
            if attempts >= MAX_ATTEMPTS_BEFORE_BACKTRACK and mastery < TEACH_MASTERY_THRESHOLD:
                target = self._find_backtrack_target(concept_id)
                if target is not None:
                    return target

        # Normal frontier selection: shallowest first (topological order)
        frontier = self.get_frontier()
        if not frontier:
            return None

        # Sort by topological depth (shallowest first)
        frontier.sort(key=lambda c: self.graph.get_topological_depth(c))
        return frontier[0]

    def is_complete(self) -> bool:
        """Check if all concepts have been mastered."""
        return all(
            self.model.predict_mastery(cid) >= TEACH_MASTERY_THRESHOLD
            for cid in self.graph.get_all_concept_ids()
        )

    def is_stuck(self) -> bool:
        """Check if the student is stuck (no frontier and not complete)."""
        return not self.is_complete() and not self.get_frontier()

    def build_context(self, concept_id: str) -> TeachingContext:
        """Build the structured context that gets passed to the LLM."""
        node = self.graph.get_node(concept_id)
        prereqs = self.graph.get_prerequisites(concept_id)

        # Check if this is a backtrack
        attempts = self.model.get_attempt_count(concept_id)
        is_backtrack = (
            self.model.predict_mastery(concept_id) >= PREREQ_MASTERY_THRESHOLD
            and attempts > 0
        )

        # Build curriculum overview
        curriculum_lines = []
        all_ids = self.graph.get_all_concept_ids()
        for cid in all_ids:
            n = self.graph.get_node(cid)
            m = self.model.predict_mastery(cid)
            if cid == concept_id:
                marker = ">> "
                status = "CURRENT"
            elif m >= PREREQ_MASTERY_THRESHOLD:
                marker = "   "
                status = "mastered"
            elif m >= TEACH_MASTERY_THRESHOLD:
                marker = "   "
                status = "in progress"
            else:
                marker = "   "
                status = "upcoming"
            title = f" - {n.lesson_title}" if n.lesson_title else ""
            curriculum_lines.append(f"{marker}{n.name}{title} ({status}, {m:.0%})")

        # Find what's next after the current concept
        next_id = self.select_next_concept()
        next_line = ""
        if next_id and next_id != concept_id:
            next_node = self.graph.get_node(next_id)
            next_line = f"\nNext up after this: {next_node.name}"

        curriculum_overview = "FULL CURRICULUM:\n" + "\n".join(curriculum_lines) + next_line

        return TeachingContext(
            concept_id=concept_id,
            concept_name=node.name,
            description=node.description,
            teaching_hints=node.teaching_hints,
            prerequisite_mastery={
                p: self.model.predict_mastery(p) for p in prereqs
            },
            student_mastery=self.model.predict_mastery(concept_id),
            attempt_number=attempts + 1,
            is_backtrack=is_backtrack,
            curriculum_overview=curriculum_overview,
        )
