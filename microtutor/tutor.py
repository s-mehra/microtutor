"""Conversational layer using Claude API for teaching and assessment."""

from __future__ import annotations

import os

import anthropic

from microtutor.planner import TeachingContext

DEFAULT_MODEL = "claude-sonnet-4-6"
MAX_HISTORY_MESSAGES = 10

PREMISE_CHECK_PROMPT = """\
You are a tutor about to teach new material that builds on a prerequisite. \
Naturally weave in a quick check of the prerequisite, the way a real \
instructor would: "Before we get into this, recall that..." followed by a \
short question. Do not announce that you are checking prerequisites.

PREREQUISITE CONCEPT: {prereq_name}
DESCRIPTION: {prereq_description}

Keep it to 2-3 sentences total. One sentence of context, one question. \
No labels, no preamble, no "Here's a question for you."
"""

TEACH_SYSTEM_PROMPT = """\
You are an experienced tutor giving a lesson. Your primary job is to EXPLAIN \
and TEACH. You are the expert. The student is here to learn from you, not to \
be interrogated.

CONCEPT: {concept_name}
DESCRIPTION: {description}

{curriculum_context}
{prereq_context}
{hints_context}
{key_topics_context}
{lesson_history_context}
{attempt_context}

TEACHING APPROACH:
Your default mode is instruction. Explain the concept clearly, give the \
formal definition, work through a concrete example. Most of your responses \
should be you teaching, not you asking. A good ratio is roughly 3 teaching \
responses for every 1 question.

When you DO ask a question, it should be because the student needs to \
actively work through something to understand it (e.g., "What do you get \
if you multiply [1,2,3] by [4,5,6] element-wise?"). Do not ask vague \
comprehension checks like "Does that make sense?" or "What do you think \
about that?"

If the student answers a question, give them clear feedback on whether \
they were right or wrong and WHY, then continue teaching the next piece. \
Do not respond to their answer with another question. Acknowledge, \
correct if needed, then move forward.

GUIDELINES:
- Every concept needs both its formal definition and an intuitive example. \
You can lead with either one, but always include both. For a math concept \
like the dot product, the student must see the actual formula or procedure \
(multiply corresponding elements, sum the products), not just an analogy. \
Analogies build intuition. Definitions build precision. Both are required.
- Build on what the student says. Reference their previous answers when \
relevant. If they showed understanding of a prerequisite, connect it: \
"Since you already know how dot products work, a weighted sum is just..."
- Keep each response under 200 words.
- No emoji. Write in prose, not bullet-point lists.
- Sound like a person who teaches this for a living.
- When explaining a structure, process, or data flow, draw an ASCII diagram \
in a fenced code block. Diagram rules:
  * Use box-drawing characters for structure: lines (─ │), corners (┌ ┐ └ ┘), \
    tees (├ ┤ ┬ ┴), and crosses (┼). Not +, -, or |.
  * Use arrows for flow: → ← ↑ ↓ for direction. Use ──→ for longer connections.
  * Label every box or node. No unlabeled shapes.
  * Pick one direction: top-to-bottom for hierarchies, left-to-right for \
    data flow and pipelines. Do not mix directions.
  * Keep diagrams under 10 lines tall and 60 characters wide.
  * Align elements carefully. Every row should have consistent spacing. \
    Count characters to make sure boxes line up.
  * For math layouts (matrices, vectors), show the actual numbers in a \
    clean grid. Use brackets: [ 1  2  3 ].
  * Include a one-line caption above or below if the diagram shows a \
    specific example (e.g., "2-layer network with 3 inputs").
  * VERIFY the diagram renders correctly: check that every opening bracket \
    has a closing bracket, every ┌ has a matching ┘, every line connects \
    to something. Broken diagrams are worse than no diagram.
- Use **bold** for key terms the first time you introduce them. Use \
`backticks` for variable names and math expressions.
- Your output will be rendered with a markdown renderer.
- IMPORTANT: Verify your examples and numbers before stating them. Never \
self-correct mid-sentence ("wait, that's wrong" or "actually, let me \
fix that"). If you use a number, make sure it is realistic and accurate. \
A tutor who second-guesses themselves in front of the student loses \
credibility.
- Do not stack analogies. One analogy per explanation is enough. If the \
student doesn't understand the first analogy, try a different one. If two \
analogies fail, switch to the formal definition and work through it \
mechanically before trying more analogies.
"""

ASSESS_SYSTEM_PROMPT = """\
Based on the lesson so far on "{concept_name}", ask the student ONE question \
that requires applying the concept. It should have a clear correct answer. \
Just ask the question directly, no lead-in.
"""

EVALUATE_TOOL = {
    "name": "evaluate_response",
    "description": "Evaluate whether the student's response demonstrates understanding.",
    "input_schema": {
        "type": "object",
        "properties": {
            "understood": {
                "type": "boolean",
                "description": "Whether the student demonstrated understanding.",
            },
            "note": {
                "type": "string",
                "description": "Brief note on what the student got right or wrong.",
            },
        },
        "required": ["understood", "note"],
        "additionalProperties": False,
    },
}

ASSESS_TOOL = {
    "name": "assess_student",
    "description": "Record the assessment result after the student answers an assessment question.",
    "input_schema": {
        "type": "object",
        "properties": {
            "correct": {
                "type": "boolean",
                "description": "Whether the student demonstrated genuine understanding of the concept.",
            },
            "misconception": {
                "type": "string",
                "description": "If incorrect, describe the specific misconception. If correct, write 'none'.",
            },
            "explanation": {
                "type": "string",
                "description": "Brief explanation of why the answer was judged correct or incorrect.",
            },
            "confidence": {
                "type": "string",
                "enum": ["clear_pass", "clear_fail", "uncertain"],
                "description": "How confident you are in the assessment. 'uncertain' means you need to ask another question to be sure.",
            },
        },
        "required": ["correct", "misconception", "explanation", "confidence"],
        "additionalProperties": False,
    },
}

FOLLOWUP_ASSESS_PROMPT = """\
You just assessed a student on "{concept_name}" but you're not fully confident \
in the result. Ask ONE more question that probes a different angle of the same \
concept. Don't repeat the same type of question. Just ask it directly.
"""

SUMMARIZE_LESSON_PROMPT = """\
Summarize the tutoring lesson that just happened on "{concept_name}". \
Review the conversation and extract key information for future reference. \
Use the record_lesson_summary tool.
"""

SUMMARIZE_TOOL = {
    "name": "record_lesson_summary",
    "description": "Record a structured summary of a completed lesson.",
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "2-3 sentence summary of what was taught and how the student performed.",
            },
            "examples_used": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of specific examples or analogies used during the lesson.",
            },
            "key_topics_covered": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Which key topics from the concept were actually covered.",
            },
            "conversation_digest": {
                "type": "string",
                "description": "Brief digest of the conversation flow for context in future lessons.",
            },
        },
        "required": ["summary", "examples_used", "key_topics_covered", "conversation_digest"],
        "additionalProperties": False,
    },
}


class Tutor:
    """Manages teaching conversations and assessments via Claude API."""

    def __init__(self) -> None:
        self.client = anthropic.Anthropic()
        self.model = os.environ.get("MICROTUTOR_MODEL", DEFAULT_MODEL)
        self.conversation_history: list[dict] = []
        self._summary: str = ""

    def _get_active_history(self) -> list[dict]:
        """Return conversation history with sliding window."""
        if len(self.conversation_history) <= MAX_HISTORY_MESSAGES:
            return list(self.conversation_history)

        recent = self.conversation_history[-MAX_HISTORY_MESSAGES:]
        if self._summary:
            summary_msg = {
                "role": "user",
                "content": f"[Session context: {self._summary}]",
            }
            return [summary_msg] + recent
        return recent

    def _append_exchange(self, user_msg: str, assistant_msg: str) -> None:
        """Append a user/assistant exchange to history."""
        self.conversation_history.append({"role": "user", "content": user_msg})
        self.conversation_history.append({"role": "assistant", "content": assistant_msg})

    def _build_teach_system(self, context: TeachingContext) -> str:
        prereq_lines = []
        for pid, mastery in context.prerequisite_mastery.items():
            level = "strong" if mastery > 0.8 else "moderate" if mastery > 0.5 else "weak"
            prereq_lines.append(f"  - {pid}: {level} ({mastery:.0%})")

        prereq_context = ""
        if prereq_lines:
            prereq_context = (
                "STUDENT'S PREREQUISITE KNOWLEDGE:\n"
                + "\n".join(prereq_lines)
            )

        hints_context = ""
        if context.teaching_hints:
            hints_context = (
                "TEACHING HINTS:\n"
                + "\n".join(f"  - {h}" for h in context.teaching_hints)
            )

        attempt_context = ""
        if context.attempt_number > 1:
            attempt_context = (
                f"NOTE: This is attempt #{context.attempt_number} at this concept. "
                f"Current mastery: {context.student_mastery:.0%}. "
                "Try a different angle or simpler explanation."
            )
        if context.is_backtrack:
            attempt_context = (
                "NOTE: The student is being re-assessed on this prerequisite "
                "because they struggled with a downstream concept. Focus on "
                "checking and reinforcing their understanding."
            )

        key_topics_context = ""
        if context.key_topics:
            key_topics_context = (
                "KEY TOPICS TO COVER IN THIS LESSON:\n"
                + "\n".join(f"  - {t}" for t in context.key_topics)
                + "\nCover all key topics before moving to assessment. "
                "The assessment should verify the student understands each key topic."
            )

        lesson_history_context = context.lesson_history_context or ""

        return TEACH_SYSTEM_PROMPT.format(
            concept_name=context.concept_name,
            description=context.description,
            curriculum_context=context.curriculum_overview,
            prereq_context=prereq_context,
            hints_context=hints_context,
            key_topics_context=key_topics_context,
            lesson_history_context=lesson_history_context,
            attempt_context=attempt_context,
        )

    def check_premise(self, prereq_name: str, prereq_description: str) -> str:
        """Ask a quick recall question about a prerequisite concept."""
        self.conversation_history = []
        self._summary = ""

        system = PREMISE_CHECK_PROMPT.format(
            prereq_name=prereq_name,
            prereq_description=prereq_description,
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=200,
            system=system,
            messages=[{
                "role": "user",
                "content": "I'm here, go ahead.",
            }],
        )

        question = response.content[0].text
        self._append_exchange("I'm ready for the prerequisite check.", question)
        return question

    def evaluate_response(self, concept_name: str, student_answer: str) -> dict:
        """Evaluate a student's response (for premise checks and understanding checks).

        Returns dict with keys: understood (bool), note (str).
        """
        self.conversation_history.append({"role": "user", "content": student_answer})

        system = (
            f"You are evaluating a student's response about '{concept_name}'. "
            "Use the evaluate_response tool to record whether they demonstrated understanding."
        )

        messages = self._get_active_history()

        response = self.client.messages.create(
            model=self.model,
            max_tokens=200,
            system=system,
            messages=messages,
            tools=[EVALUATE_TOOL],
            tool_choice={"type": "tool", "name": "evaluate_response"},
        )

        for block in response.content:
            if block.type == "tool_use" and block.name == "evaluate_response":
                result = block.input
                feedback = "Good." if result["understood"] else result["note"]
                self.conversation_history.append({
                    "role": "assistant",
                    "content": feedback,
                })
                return result

        return {"understood": False, "note": "Could not evaluate response."}

    def start_teaching_stream(self, context: TeachingContext):
        """Start the lesson, returning a streaming context manager.

        Appends user message to history. Caller must consume the stream's
        text_stream iterator, then call record_response(text).
        """
        system = self._build_teach_system(context)
        user_msg = f"Go ahead, I'm ready to learn about {context.concept_name}."
        self.conversation_history.append({"role": "user", "content": user_msg})

        return self.client.messages.stream(
            model=self.model,
            max_tokens=600,
            system=system,
            messages=self._get_active_history(),
        )

    def continue_teaching_stream(self, context: TeachingContext, student_message: str):
        """Continue teaching, returning a streaming context manager.

        Appends user message to history. Caller must consume the stream's
        text_stream iterator, then call record_response(text, context).
        """
        self.conversation_history.append({
            "role": "user",
            "content": student_message,
        })

        system = self._build_teach_system(context)

        return self.client.messages.stream(
            model=self.model,
            max_tokens=500,
            system=system,
            messages=self._get_active_history(),
        )

    def record_response(self, text: str, context: TeachingContext | None = None) -> None:
        """Record an assistant response in conversation history after streaming."""
        self.conversation_history.append({
            "role": "assistant",
            "content": text,
        })

        if context and len(self.conversation_history) > MAX_HISTORY_MESSAGES:
            older = self.conversation_history[:-MAX_HISTORY_MESSAGES]
            self._summary = (
                f"Teaching {context.concept_name}. "
                f"{len(older)} earlier exchanges covered initial explanation and Q&A."
            )

    def start_teaching(self, context: TeachingContext) -> str:
        """Start the lesson (non-streaming). Returns completed text."""
        with self.start_teaching_stream(context) as stream:
            response = stream.get_final_message()
        text = response.content[0].text
        self.record_response(text)
        return text

    def continue_teaching(self, context: TeachingContext, student_message: str) -> str:
        """Continue teaching (non-streaming). Returns completed text."""
        with self.continue_teaching_stream(context, student_message) as stream:
            response = stream.get_final_message()
        text = response.content[0].text
        self.record_response(text, context)
        return text

    def ask_assessment_question(self, context: TeachingContext) -> str:
        """Ask an assessment question. Returns the question text."""
        system = ASSESS_SYSTEM_PROMPT.format(concept_name=context.concept_name)

        messages = self._get_active_history()
        messages.append({
            "role": "user",
            "content": "Go ahead.",
        })

        response = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            system=system,
            messages=messages,
        )

        question_text = response.content[0].text
        self._append_exchange("Go ahead.", question_text)
        return question_text

    def judge_answer(self, context: TeachingContext, student_answer: str) -> dict:
        """Judge the student's answer to an assessment question.

        Returns dict with keys: correct (bool), misconception (str), explanation (str).
        """
        self.conversation_history.append({
            "role": "user",
            "content": student_answer,
        })

        system = (
            f"You are assessing a student's answer about '{context.concept_name}'. "
            f"Concept: {context.description}. "
            "Use the assess_student tool to record your judgment."
        )

        messages = self._get_active_history()

        response = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            system=system,
            messages=messages,
            tools=[ASSESS_TOOL],
            tool_choice={"type": "tool", "name": "assess_student"},
        )

        for block in response.content:
            if block.type == "tool_use" and block.name == "assess_student":
                result = block.input
                self.conversation_history.append({
                    "role": "assistant",
                    "content": f"Assessment: {'Correct' if result['correct'] else 'Incorrect'}. {result['explanation']}",
                })
                return result

        return {
            "correct": False,
            "misconception": "Assessment failed",
            "explanation": "Could not determine assessment result.",
            "confidence": "clear_fail",
        }

    def ask_followup_question(self, context: TeachingContext) -> str:
        """Ask a followup assessment question from a different angle."""
        system = FOLLOWUP_ASSESS_PROMPT.format(concept_name=context.concept_name)

        messages = self._get_active_history()
        messages.append({
            "role": "user",
            "content": "Go on.",
        })

        response = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            system=system,
            messages=messages,
        )

        question_text = response.content[0].text
        self._append_exchange("Go on.", question_text)
        return question_text

    def summarize_lesson(self, concept_name: str) -> dict:
        """Summarize the lesson that just completed using the conversation history.

        Returns dict with: summary, examples_used, key_topics_covered, conversation_digest.
        """
        system = SUMMARIZE_LESSON_PROMPT.format(concept_name=concept_name)
        messages = self._get_active_history()

        response = self.client.messages.create(
            model=self.model,
            max_tokens=400,
            system=system,
            messages=messages,
            tools=[SUMMARIZE_TOOL],
            tool_choice={"type": "tool", "name": "record_lesson_summary"},
        )

        for block in response.content:
            if block.type == "tool_use" and block.name == "record_lesson_summary":
                return block.input

        return {
            "summary": "",
            "examples_used": [],
            "key_topics_covered": [],
            "conversation_digest": "",
        }
