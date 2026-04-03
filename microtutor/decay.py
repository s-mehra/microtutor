"""Knowledge decay: mastery scores decrease over time without practice."""

from __future__ import annotations

import time

from microtutor.model import StudentModel

DEFAULT_HALF_LIFE_DAYS = 14.0
MASTERY_FLOOR = 0.05  # never decay below p_init
DECAY_REPORT_THRESHOLD = 0.05  # only report decays larger than this


def apply_decay(
    mastery: float,
    last_updated_at: float,
    current_time: float | None = None,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    floor: float = MASTERY_FLOOR,
) -> float:
    """Apply exponential decay to a mastery value.

    Formula: floor + (mastery - floor) * 0.5 ^ (elapsed_days / half_life_days)

    Returns the decayed mastery, never below floor.
    """
    if current_time is None:
        current_time = time.time()

    if last_updated_at <= 0 or mastery <= floor:
        return mastery

    elapsed_seconds = current_time - last_updated_at
    if elapsed_seconds <= 0:
        return mastery

    elapsed_days = elapsed_seconds / 86400.0
    decay_factor = 0.5 ** (elapsed_days / half_life_days)
    decayed = floor + (mastery - floor) * decay_factor

    return max(floor, decayed)


def apply_decay_to_model(
    model: StudentModel,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    current_time: float | None = None,
) -> dict[str, float]:
    """Apply decay to all concepts in a student model.

    Mutates mastery values in place.
    Returns dict of {concept_id: decay_amount} for concepts that decayed
    by more than DECAY_REPORT_THRESHOLD.
    """
    if current_time is None:
        current_time = time.time()

    decayed_concepts = {}

    for concept_id, state in model.states.items():
        old_mastery = state.mastery
        new_mastery = apply_decay(
            mastery=old_mastery,
            last_updated_at=state.last_updated_at,
            current_time=current_time,
            half_life_days=half_life_days,
        )

        decay_amount = old_mastery - new_mastery
        if decay_amount > DECAY_REPORT_THRESHOLD:
            decayed_concepts[concept_id] = decay_amount

        state.mastery = new_mastery

    return decayed_concepts
