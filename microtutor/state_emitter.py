"""Builds GraphState snapshots for the visualization."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from microtutor.graph import ConceptGraph
from microtutor.model import StudentModel
from microtutor.planner import PREREQ_MASTERY_THRESHOLD, TEACH_MASTERY_THRESHOLD


@dataclass
class ConceptVizState:
    id: str
    name: str
    mastery: float
    attempts: int
    status: str  # "mastered" | "in_progress" | "current" | "frontier" | "locked"
    depth: int
    last_updated_at: float


@dataclass
class GraphState:
    """Snapshot of the knowledge graph state."""

    course_title: str
    current_concept: str | None
    concepts: list[ConceptVizState]
    edges: list[tuple[str, str]]  # (source_id, target_id)
    frontier: list[str]
    total_mastered: int
    total_concepts: int
    timestamp: float = field(default_factory=time.time)


class StateEmitter:
    """Builds GraphState snapshots from the teaching engine."""

    def build_snapshot(
        self,
        model: StudentModel,
        graph: ConceptGraph,
        current_concept: str | None,
        course_title: str,
    ) -> GraphState:
        """Build an immutable snapshot of the current graph state."""
        all_ids = graph.get_all_concept_ids()

        # Compute frontier
        frontier = []
        for cid in all_ids:
            mastery = model.predict_mastery(cid)
            if mastery >= TEACH_MASTERY_THRESHOLD:
                continue
            prereqs = graph.get_prerequisites(cid)
            if all(model.predict_mastery(p) >= PREREQ_MASTERY_THRESHOLD for p in prereqs):
                frontier.append(cid)

        concepts = []
        total_mastered = 0
        for cid in all_ids:
            node = graph.get_node(cid)
            state = model.states[cid]
            mastery = state.mastery

            if cid == current_concept:
                status = "current"
            elif mastery >= PREREQ_MASTERY_THRESHOLD:
                status = "mastered"
                total_mastered += 1
            elif mastery >= TEACH_MASTERY_THRESHOLD:
                status = "in_progress"
            elif cid in frontier:
                status = "frontier"
            else:
                status = "locked"

            concepts.append(ConceptVizState(
                id=cid,
                name=node.name,
                mastery=mastery,
                attempts=state.attempts,
                status=status,
                depth=graph.get_topological_depth(cid),
                last_updated_at=state.last_updated_at,
            ))

        edges = [
            (src, tgt)
            for tgt in all_ids
            for src in graph.get_prerequisites(tgt)
        ]

        return GraphState(
            course_title=course_title,
            current_concept=current_concept,
            concepts=concepts,
            edges=edges,
            frontier=frontier,
            total_mastered=total_mastered,
            total_concepts=len(all_ids),
        )
