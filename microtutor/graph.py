"""Concept graph for representing knowledge domains as a prerequisite DAG."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx


@dataclass
class ConceptNode:
    id: str
    name: str
    description: str
    lesson_title: str = ""
    teaching_hints: list[str] = field(default_factory=list)
    key_topics: list[str] = field(default_factory=list)
    bkt_params: dict[str, float] = field(default_factory=lambda: {
        "p_init": 0.05,
        "p_learn": 0.1,
        "p_guess": 0.2,
        "p_slip": 0.1,
    })


class ConceptGraph:
    """A directed acyclic graph of concepts with prerequisite edges."""

    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self._nodes: dict[str, ConceptNode] = {}

    @classmethod
    def from_dict(cls, data: dict) -> ConceptGraph:
        """Build a ConceptGraph from a dictionary (same shape as the JSON file)."""
        cg = cls()

        for node_data in data["concepts"]:
            node = ConceptNode(
                id=node_data["id"],
                name=node_data["name"],
                description=node_data["description"],
                lesson_title=node_data.get("lesson_title", ""),
                teaching_hints=node_data.get("teaching_hints", []),
                key_topics=node_data.get("key_topics", []),
                bkt_params=node_data.get("bkt_params", {
                    "p_init": 0.05,
                    "p_learn": 0.1,
                    "p_guess": 0.2,
                    "p_slip": 0.1,
                }),
            )
            cg._nodes[node.id] = node
            cg.graph.add_node(node.id)

        for node_data in data["concepts"]:
            for prereq_id in node_data.get("prerequisites", []):
                if prereq_id not in cg._nodes:
                    raise ValueError(
                        f"Prerequisite '{prereq_id}' for concept "
                        f"'{node_data['id']}' not found in graph"
                    )
                cg.graph.add_edge(prereq_id, node_data["id"])

        if not nx.is_directed_acyclic_graph(cg.graph):
            raise ValueError("Concept graph contains cycles")

        return cg

    @classmethod
    def from_json(cls, path: str | Path) -> ConceptGraph:
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def get_node(self, concept_id: str) -> ConceptNode:
        return self._nodes[concept_id]

    def get_prerequisites(self, concept_id: str) -> list[str]:
        return list(self.graph.predecessors(concept_id))

    def get_all_concept_ids(self) -> list[str]:
        return list(nx.topological_sort(self.graph))

    def get_topological_depth(self, concept_id: str) -> int:
        """Return the length of the longest path from any root to this node."""
        # BFS-based longest path in DAG using topological order
        depths: dict[str, int] = {}
        for node in nx.topological_sort(self.graph):
            preds = list(self.graph.predecessors(node))
            if not preds:
                depths[node] = 0
            else:
                depths[node] = max(depths[p] + 1 for p in preds)
        return depths.get(concept_id, 0)
