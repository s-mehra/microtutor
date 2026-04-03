"""Student model using Bayesian Knowledge Tracing for mastery estimation."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from microtutor.graph import ConceptGraph

SCHEMA_VERSION = 1


@dataclass
class ConceptState:
    mastery: float  # P(L_n) - probability the student has learned the concept
    attempts: int = 0


class StudentModel:
    """Tracks a student's mastery of concepts using Bayesian Knowledge Tracing.

    BKT update equations (Corbett & Anderson, 1995):
        P(L_n | correct) = P(L_n) * (1 - p_slip) / P(correct)
        P(L_n | incorrect) = P(L_n) * p_slip / P(incorrect)
        P(L_{n+1}) = P(L_n | obs) + (1 - P(L_n | obs)) * p_learn
    where:
        P(correct) = P(L_n) * (1 - p_slip) + (1 - P(L_n)) * p_guess
        P(incorrect) = 1 - P(correct)
    """

    def __init__(self, graph: ConceptGraph) -> None:
        self.graph = graph
        self.states: dict[str, ConceptState] = {}
        self._observation_log: list[dict] = []

        for concept_id in graph.get_all_concept_ids():
            node = graph.get_node(concept_id)
            p_init = node.bkt_params.get("p_init", 0.05)
            self.states[concept_id] = ConceptState(mastery=p_init)

    def predict_mastery(self, concept_id: str) -> float:
        return self.states[concept_id].mastery

    def update(self, concept_id: str, correct: bool) -> float:
        """Update mastery estimate based on an observation. Returns new mastery."""
        state = self.states[concept_id]
        node = self.graph.get_node(concept_id)
        params = node.bkt_params

        p_l = state.mastery
        p_guess = params.get("p_guess", 0.2)
        p_slip = params.get("p_slip", 0.1)
        p_learn = params.get("p_learn", 0.1)

        # P(correct) given current mastery
        p_correct = p_l * (1 - p_slip) + (1 - p_l) * p_guess

        # Posterior: P(L_n | observation)
        if correct:
            p_l_given_obs = (p_l * (1 - p_slip)) / p_correct
        else:
            p_incorrect = 1 - p_correct
            # Guard against division by zero (shouldn't happen with valid params)
            if p_incorrect < 1e-10:
                p_l_given_obs = p_l
            else:
                p_l_given_obs = (p_l * p_slip) / p_incorrect

        # Learning transition: P(L_{n+1})
        new_mastery = p_l_given_obs + (1 - p_l_given_obs) * p_learn

        # Clamp to [0, 1] to handle floating point edge cases
        state.mastery = max(0.0, min(1.0, new_mastery))
        state.attempts += 1

        # Log observation for future parameter fitting
        self._observation_log.append({
            "concept_id": concept_id,
            "correct": correct,
            "mastery_before": p_l,
            "mastery_after": new_mastery,
            "timestamp": time.time(),
        })

        return new_mastery

    def partial_update(self, concept_id: str, understood: bool) -> float:
        """Softer mastery update for mid-lesson signals (premise checks, understanding checks).

        Uses a reduced learning rate to move mastery less aggressively than a
        full assessment update.
        """
        state = self.states[concept_id]
        p_l = state.mastery

        if understood:
            # Gentle boost: move 30% of the way toward confident
            new_mastery = p_l + (1 - p_l) * 0.05
        else:
            # Gentle reduction: move 30% of the way toward uncertain
            new_mastery = p_l * 0.9

        state.mastery = max(0.0, min(1.0, new_mastery))

        self._observation_log.append({
            "concept_id": concept_id,
            "correct": understood,
            "mastery_before": p_l,
            "mastery_after": new_mastery,
            "timestamp": time.time(),
            "type": "partial",
        })

        return new_mastery

    def get_attempt_count(self, concept_id: str) -> int:
        return self.states[concept_id].attempts

    def save(self, path: str | Path) -> None:
        path = Path(path)
        data = {
            "schema_version": SCHEMA_VERSION,
            "states": {
                cid: {"mastery": s.mastery, "attempts": s.attempts}
                for cid, s in self.states.items()
            },
            "observation_log": self._observation_log,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            return  # Fresh student, keep initial state

        with open(path) as f:
            data = json.load(f)

        version = data.get("schema_version", 0)
        if version != SCHEMA_VERSION:
            # Future: run migration functions here
            return

        for cid, state_data in data.get("states", {}).items():
            if cid in self.states:
                self.states[cid].mastery = state_data["mastery"]
                self.states[cid].attempts = state_data["attempts"]

        self._observation_log = data.get("observation_log", [])

    def save_observation_log(self, path: str | Path) -> None:
        """Append observations to a JSONL file for future parameter fitting."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            for obs in self._observation_log:
                f.write(json.dumps(obs) + "\n")
