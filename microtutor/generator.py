"""Goal-based course generation using Claude Opus 4.6."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import anthropic

from microtutor.graph import ConceptGraph

GENERATOR_MODEL = "claude-opus-4-6"

GOAL_CONVERSATION_SYSTEM = """\
You are a curriculum designer and diagnostic interviewer for an AI tutoring \
system. This conversation is the most important step in building a personalized \
course. Your job is to understand three things deeply: what the student wants \
to achieve, what they actually know right now, and where the boundary is \
between what they know and what they don't.

This conversation IS the first assessment. Treat it as a diagnostic, not an \
intake form.

CONVERSATION PHASES:

Phase 1 — TOPIC AND MOTIVATION (1-2 exchanges)
Ask what they want to learn and WHY. The "why" shapes everything. "I want to \
learn ML" means different things for a data scientist evaluating vendor models \
vs a student who wants to publish a paper vs someone building a product. \
Get the specific use case.

Phase 2 — KNOWLEDGE PROBING (3-5 exchanges)
This is where you spend the most time. Do NOT accept vague claims like "I know \
some math" or "I'm familiar with Python." Probe with specific questions:
- "Can you explain what a derivative tells you, in your own words?"
- "If I gave you two vectors, could you compute their dot product?"
- "What happens when you multiply a matrix by a vector?"
- "What's the difference between supervised and unsupervised learning?"

Adapt your probing based on the topic. If they claim to know something, test \
it with a concrete question. If they get it right, probe deeper. If they get \
it wrong or are vague, you've found the boundary — note that and move on.

You're mapping the frontier: what they genuinely understand vs what they've \
only heard of vs what's completely new. This directly determines which concepts \
start with high mastery (skip or light review) vs which need full teaching.

Phase 3 — LEARNING CONTEXT (1-2 exchanges)
How much time do they have? Are they building something specific or learning \
for its own sake? Do they want theoretical depth or practical ability? Do they \
learn better from examples first or definitions first?

Phase 4 — FINALIZE
When you have a clear picture, use the finalize_goals tool. Do NOT finalize \
early. A shallow diagnostic produces a generic course. Take 6-10 exchanges \
total.

RULES:
- One question per message. Never batch questions.
- React to their answers. If they reveal something surprising, follow up.
- Be conversational, not clinical. Sound like a tutor, not a form.
- When probing knowledge, give brief feedback: "Good, that's right" or "Not \
  quite — that's actually [correction]. We'll cover that."
- Track every specific knowledge claim and test result mentally. These feed \
  directly into the curriculum.
"""

FINALIZE_GOALS_TOOL = {
    "name": "finalize_goals",
    "description": "Finalize the diagnostic into a structured course brief.",
    "input_schema": {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "The main subject area.",
            },
            "goals": {
                "type": "string",
                "description": "What the student wants to be able to DO after the course. Specific and actionable.",
            },
            "background": {
                "type": "string",
                "description": "Detailed assessment of what the student knows. Include specific concepts they demonstrated understanding of, concepts they were vague on, and concepts they did not know.",
            },
            "knowledge_frontier": {
                "type": "string",
                "description": "The boundary between what they know and don't know. E.g., 'Solid on single-variable calculus and basic linear algebra. Understands what a matrix is but cannot do multiplication. No exposure to probability.'",
            },
            "desired_depth": {
                "type": "string",
                "enum": ["intro", "intermediate", "advanced"],
                "description": "How deep the course should go.",
            },
            "learning_context": {
                "type": "string",
                "description": "Practical vs theoretical preference, time constraints, what they're building, learning style notes.",
            },
            "course_title": {
                "type": "string",
                "description": "A concise, specific title for the course.",
            },
            "course_description": {
                "type": "string",
                "description": "A 1-2 sentence description of the course tailored to this student.",
            },
        },
        "required": [
            "topic", "goals", "background", "knowledge_frontier",
            "desired_depth", "learning_context", "course_title", "course_description",
        ],
        "additionalProperties": False,
    },
}

GRAPH_GENERATION_PROMPT = """\
Generate a knowledge graph for a personalized tutoring course. This graph \
must be tailored to the specific student based on the diagnostic below.

STUDENT DIAGNOSTIC:
Topic: {topic}
Goals: {goals}
Background assessment: {background}
Knowledge frontier: {knowledge_frontier}
Desired depth: {desired_depth}
Learning context: {learning_context}

FULL DIAGNOSTIC CONVERSATION:
{conversation_text}

Generate a concept graph as a JSON object. The graph should:

1. START where the student's knowledge ENDS. If they demonstrated understanding \
of vectors, don't include a "what is a vector" node. If they were vague on \
matrix multiplication, include it but set p_init to 0.3 (partial knowledge). \
If something is completely new, p_init stays at 0.05 (default).

2. END where the student's goals are met. Don't keep going past what they need.

3. ADAPT to their learning context. If they're building a product, emphasize \
practical application. If they want theoretical depth, include proofs and \
derivations.

JSON structure:
{{
  "title": "{course_title}",
  "description": "{course_description}",
  "concepts": [
    {{
      "id": "concept_id",
      "name": "Concept Name",
      "lesson_title": "Short Evocative Title",
      "description": "What this concept is and why it matters for THIS student",
      "prerequisites": ["prereq_id"],
      "teaching_hints": [
        "hint that references the student's background or goals",
        "specific pedagogical approach for this concept"
      ],
      "key_topics": [
        "specific sub-topic that must be covered",
        "another sub-topic",
        "sub-topics should be things the student needs, not generic textbook sections"
      ],
      "bkt_params": {{
        "p_init": 0.05,
        "p_learn": 0.1,
        "p_guess": 0.2,
        "p_slip": 0.1
      }}
    }}
  ]
}}

p_init calibration based on diagnostic:
- 0.05: Student has never encountered this concept
- 0.15-0.25: Student has heard of it but couldn't explain it
- 0.3-0.5: Student was vague or partially correct when probed
- 0.6-0.7: Student demonstrated solid understanding but hasn't practiced recently
- 0.8+: Do not include this concept. The student already knows it. Skip it.

Requirements:
- 8-25 concept nodes depending on scope
- Prerequisites form a DAG (no cycles). Order concepts so prerequisites come first.
- Each concept needs 3-8 key_topics that are specific and actionable
- Each concept needs 2-3 teaching_hints (at least one should reference the student's goals or background)
- lesson_title should be short and evocative
- key_topics should reflect what THIS student needs for THIS concept, not a generic textbook TOC
- teaching_hints should leverage what you learned about this student in the diagnostic

Output ONLY the JSON object. No markdown fencing. No explanation.
"""


@dataclass
class GoalBrief:
    topic: str
    goals: str
    background: str
    knowledge_frontier: str
    desired_depth: str
    learning_context: str
    course_title: str
    course_description: str
    conversation: list[dict] = field(default_factory=list)


class CourseGenerator:
    """Runs the goal-setting conversation and generates the knowledge graph."""

    def __init__(self, api_key: str) -> None:
        self.client = anthropic.Anthropic(api_key=api_key)
        self.conversation: list[dict] = []

    def get_next_message(self, student_message: str | None = None) -> str | GoalBrief:
        """Continue the goal conversation.

        Pass None for the first call (tutor opens).
        Returns a string (tutor's next question) or a GoalBrief (conversation complete).
        """
        if student_message is not None:
            self.conversation.append({"role": "user", "content": student_message})

        response = self.client.messages.create(
            model=GENERATOR_MODEL,
            max_tokens=500,
            system=GOAL_CONVERSATION_SYSTEM,
            messages=self.conversation or [{"role": "user", "content": "I want to learn something new."}],
            tools=[FINALIZE_GOALS_TOOL],
        )

        # Check if Claude used the finalize tool
        for block in response.content:
            if block.type == "tool_use" and block.name == "finalize_goals":
                brief = GoalBrief(
                    topic=block.input["topic"],
                    goals=block.input["goals"],
                    background=block.input["background"],
                    knowledge_frontier=block.input.get("knowledge_frontier", ""),
                    desired_depth=block.input["desired_depth"],
                    learning_context=block.input.get("learning_context", ""),
                    course_title=block.input["course_title"],
                    course_description=block.input["course_description"],
                    conversation=list(self.conversation),
                )
                return brief

        # Otherwise it's a text response (another question)
        text = ""
        for block in response.content:
            if block.type == "text":
                text = block.text
                break

        self.conversation.append({"role": "assistant", "content": text})
        return text

    def generate_graph(self, brief: GoalBrief, max_retries: int = 2) -> dict:
        """Generate a concept graph from the goal brief. Returns raw dict."""
        # Build conversation text for the generation prompt
        conv_lines = []
        for msg in brief.conversation:
            role = "Student" if msg["role"] == "user" else "Tutor"
            conv_lines.append(f"{role}: {msg['content']}")
        conversation_text = "\n".join(conv_lines) if conv_lines else "No conversation recorded."

        prompt = GRAPH_GENERATION_PROMPT.format(
            topic=brief.topic,
            background=brief.background,
            knowledge_frontier=brief.knowledge_frontier,
            goals=brief.goals,
            desired_depth=brief.desired_depth,
            learning_context=brief.learning_context,
            course_title=brief.course_title,
            course_description=brief.course_description,
            conversation_text=conversation_text,
        )

        last_error = None
        last_error_text = ""
        for attempt in range(max_retries):
            messages = [{"role": "user", "content": prompt}]

            # On retry, include the error so Claude can fix it
            if last_error:
                messages.append({"role": "assistant", "content": last_error_text})
                messages.append({
                    "role": "user",
                    "content": (
                        f"That JSON was invalid: {last_error}. "
                        "Please output the complete, valid JSON again. "
                        "Make sure it is not truncated."
                    ),
                })

            response = self.client.messages.create(
                model=GENERATOR_MODEL,
                max_tokens=16000,
                messages=messages,
            )

            # Check for truncation
            if response.stop_reason == "max_tokens":
                last_error = "Response was truncated (hit max_tokens)"
                last_error_text = response.content[0].text
                continue

            text = response.content[0].text.strip()
            text = _extract_json(text)

            try:
                return json.loads(text)
            except json.JSONDecodeError as e:
                last_error = str(e)
                last_error_text = text
                continue

        raise ValueError(f"Failed to generate valid JSON after {max_retries} attempts: {last_error}")

    def validate_graph(self, graph_data: dict) -> ConceptGraph:
        """Validate graph data through ConceptGraph. Raises on invalid."""
        return ConceptGraph.from_dict(graph_data)

    def save_graph(self, graph_data: dict, path: Path) -> None:
        """Save graph data to a JSON file."""
        with open(path, "w") as f:
            json.dump(graph_data, f, indent=2)


def _extract_json(text: str) -> str:
    """Extract JSON from LLM output, stripping markdown fencing and surrounding text."""
    # Strip markdown fencing
    if "```" in text:
        lines = text.split("\n")
        in_fence = False
        json_lines = []
        for line in lines:
            if line.strip().startswith("```") and not in_fence:
                in_fence = True
                continue
            elif line.strip() == "```" and in_fence:
                in_fence = False
                continue
            elif in_fence:
                json_lines.append(line)
        if json_lines:
            text = "\n".join(json_lines)

    # Find the outermost { ... } pair
    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    end = start
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    return text[start:end + 1]
