"""Textual TUI application for microtutor."""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path

from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Input, Static, Header

from microtutor.graph import ConceptGraph
from microtutor.model import StudentModel
from microtutor.planner import Planner
from microtutor.tutor import Tutor

CHARS_PER_SEC = 25.0
EST_MINUTES = 8


class MicrotutorApp(App):
    """Knowledge-State Aware AI Tutor."""

    TITLE = "microtutor"

    CSS = """
    Screen {
        layout: vertical;
        background: $surface;
    }

    #conversation {
        height: 1fr;
        overflow-y: auto;
        scrollbar-gutter: stable;
        padding: 1 3;
    }

    .tutor-label {
        margin: 1 0 0 2;
        color: cyan;
        text-style: bold;
    }

    .tutor-msg {
        margin: 0 2 1 4;
    }

    .student-msg {
        margin: 0 2 1 2;
    }

    .system-msg {
        margin: 0 2 1 2;
    }

    #user-input {
        dock: bottom;
        margin: 0 3 1 3;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "interrupt", "Skip / Quit", priority=True),
        Binding("escape", "save_quit", "Save & Quit"),
    ]

    def __init__(self, graph_path: Path, students_dir: Path) -> None:
        super().__init__()
        self.graph_path = graph_path
        self.students_dir = students_dir
        self.graph: ConceptGraph | None = None
        self.model: StudentModel | None = None
        self.student_path: Path | None = None
        self._input_event = asyncio.Event()
        self._input_value = ""
        self._typewriting = False
        self._skip_typewriter = False
        self._waiting_for_input = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield VerticalScroll(id="conversation")
        yield Input(
            placeholder="Type here... (type 'exit' to save and close)",
            id="user-input",
        )

    def on_mount(self) -> None:
        self.run_session()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()

        # Exit always works, regardless of state
        if value.lower() in ("quit", "exit", "q"):
            event.input.clear()
            self._save_and_quit()
            return

        if not self._waiting_for_input:
            return

        event.input.clear()
        self._input_value = value
        if value:
            self._add(Static(
                Text(f"  You > {value}", style="bold bright_white"),
                classes="student-msg",
            ))
        self._input_event.set()

    def action_interrupt(self) -> None:
        if self._typewriting:
            self._skip_typewriter = True
        else:
            self._save_and_quit()

    def action_save_quit(self) -> None:
        self._save_and_quit()

    def _save_and_quit(self) -> None:
        if self.model and self.student_path:
            self.model.save(self.student_path)
        self.exit()

    # --- Widget helpers ---

    def _add(self, widget: Static) -> None:
        container = self.query_one("#conversation", VerticalScroll)
        container.mount(widget)
        # Only auto-scroll if user is near the bottom (not scrolled up reading)
        if container.max_scroll_y == 0 or container.scroll_y >= container.max_scroll_y - 3:
            container.scroll_end(animate=False)

    def _write(self, renderable) -> None:
        self._add(Static(renderable))

    def _info(self, text: str) -> None:
        self._write(Text(f"  {text}", style="dim"))

    def _error(self, text: str) -> None:
        self._write(Text(f"  Error: {text}", style="bold red"))

    # --- Input ---

    async def _ask(self, allow_empty: bool = False) -> str | None:
        self._waiting_for_input = True
        self._input_event.clear()
        self.query_one("#user-input", Input).focus()
        await self._input_event.wait()
        self._waiting_for_input = False
        value = self._input_value
        if value.lower() in ("quit", "exit", "q"):
            return None
        if not allow_empty and not value:
            return await self._ask(allow_empty)
        return value

    # --- Tutor output ---

    def _tutor_label(self) -> None:
        self._add(Static(Text("  Tutor", style="bold cyan"), classes="tutor-label"))

    def _tutor_says(self, text: str) -> None:
        self._tutor_label()
        self._add(Static(Markdown(text), classes="tutor-msg"))

    async def _stream_tutor(self, stream_method, *args) -> str:
        """Buffer API response in a thread, then typewriter it."""
        self._tutor_label()

        accumulated = await asyncio.to_thread(self._buffer_stream, stream_method, *args)

        segments = _split_prose_and_code(accumulated)
        for seg_type, content in segments:
            if seg_type == "code":
                self._render_code_block(content)
            else:
                await self._typewriter_prose(content)

        return accumulated

    def _buffer_stream(self, stream_method, *args) -> str:
        accumulated = ""
        with stream_method(*args) as stream:
            for chunk in stream.text_stream:
                accumulated += chunk
        return accumulated

    def _smart_scroll(self, container: VerticalScroll) -> None:
        """Auto-scroll only if the user is near the bottom."""
        if container.max_scroll_y == 0 or container.scroll_y >= container.max_scroll_y - 3:
            container.scroll_end(animate=False)

    async def _typewriter_prose(self, text: str) -> None:
        if not text.strip():
            return

        container = self.query_one("#conversation", VerticalScroll)
        widget = Static("", classes="tutor-msg")
        container.mount(widget)

        self._typewriting = True
        self._skip_typewriter = False

        revealed = ""
        last_render = time.monotonic()
        delay = 1.0 / CHARS_PER_SEC

        for i, char in enumerate(text):
            if self._skip_typewriter:
                break
            revealed += char
            now = time.monotonic()
            if now - last_render >= 0.08 or i == len(text) - 1:
                widget.update(Markdown(revealed))
                self._smart_scroll(container)
                last_render = now
            await asyncio.sleep(delay)

        # Final render with full text
        widget.update(Markdown(text))
        self._smart_scroll(container)
        self._typewriting = False

    def _render_code_block(self, content: str) -> None:
        lines = content.strip().split("\n")
        lang = ""
        if lines and lines[0].startswith("```"):
            lang = lines[0][3:].strip()
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code_text = "\n".join(lines)

        if lang:
            self._write(Syntax(code_text, lang, theme="monokai", padding=1))
        else:
            self._write(Panel(
                Text(code_text, style="bright_white"),
                border_style="bright_cyan",
                padding=(0, 1),
            ))

    # --- API call wrapper ---

    async def _call(self, fn, *args, **kwargs):
        return await asyncio.to_thread(fn, *args, **kwargs)

    # --- Mastery display ---

    def _write_mastery(self) -> None:
        table = Table(
            show_header=False, show_edge=False, padding=(0, 1),
            expand=True, title="Mastery", title_style="bold white",
        )
        table.add_column("Concept", style="white", ratio=3)
        table.add_column("Bar", ratio=4)
        table.add_column("Score", justify="right", ratio=1)

        for cid in self.graph.get_all_concept_ids():
            mastery = self.model.predict_mastery(cid)
            node = self.graph.get_node(cid)
            color = "green" if mastery >= 0.7 else "yellow" if mastery >= 0.4 else "red"
            bar_len = int(mastery * 20)
            bar = Text()
            bar.append("=" * bar_len, style=f"bold {color}")
            bar.append("-" * (20 - bar_len), style="dim")
            table.add_row(node.name, bar, Text(f"{mastery:.0%}", style=f"bold {color}"))

        self._write(table)

    def _write_lesson_header(self, name: str, title: str, mastery: float) -> None:
        header = Text()
        header.append(name, style="bold bright_cyan")
        header.append(f"  -  {title}", style="white")
        meta = Text()
        meta.append(f"mastery: {mastery:.0%}", style="dim")
        meta.append(f"  |  ~{EST_MINUTES} min", style="dim")
        content = Text()
        content.append_text(header)
        content.append("\n")
        content.append_text(meta)
        self._write(Panel(content, border_style="bright_cyan", padding=(0, 2)))

    def _write_assessment(self, result: dict) -> None:
        correct = result["correct"]
        if correct:
            self._write(Text("  Correct", style="bold green"))
            self._write(Text(f"    {result['explanation']}", style="white"))
        else:
            self._write(Text("  Incorrect", style="bold red"))
            self._write(Text(f"    {result['explanation']}", style="white"))
            misconception = result.get("misconception", "none")
            if misconception and misconception != "none":
                self._write(Text(f"    {misconception}", style="dim"))

    def _write_lesson_summary(
        self, name: str, mastery: float, duration: float,
        next_name: str | None = None, next_title: str | None = None,
    ) -> None:
        mins = int(duration // 60)
        secs = int(duration % 60)
        dur_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"

        color = "green" if mastery >= 0.7 else "yellow" if mastery >= 0.4 else "red"
        content = Text()
        content.append(f"{name}: ", style="white")
        content.append(f"{mastery:.0%}", style=f"bold {color}")
        content.append(f"  |  {dur_str}", style="dim")

        if next_name:
            content.append("\n")
            content.append(f"Next: {next_name}", style="bright_cyan")
            if next_title:
                content.append(f" - {next_title}", style="white")
            content.append(f"  (~{EST_MINUTES} min)", style="dim")

        self._write(Panel(content, border_style="dim", padding=(0, 2),
                          title="Lesson Complete", title_align="left"))

    # --- Session flow ---

    @work(thread=False, exclusive=True)
    async def run_session(self) -> None:
        try:
            await self._session_flow()
        except Exception as e:
            self._error(str(e))

    async def _session_flow(self) -> None:
        # Load graph
        self.graph = ConceptGraph.from_json(self.graph_path)

        # Welcome
        self._write(Panel(
            Text.assemble(
                ("microtutor", "bold bright_cyan"), "\n",
                ("Knowledge-State Aware AI Tutor", "dim"),
            ),
            border_style="bright_cyan",
            padding=(1, 3),
        ))

        # Get name
        self._info("What's your name?")
        name = await self._ask()
        if name is None:
            self._save_and_quit()
            return

        # Load or create student
        self.students_dir.mkdir(parents=True, exist_ok=True)
        self.student_path = self.students_dir / f"{name}.json"
        self.model = StudentModel(self.graph)

        returning = self.student_path.exists()
        if returning:
            self.model.load(self.student_path)

        self._write(Panel(
            Text.assemble(
                ("microtutor", "bold bright_cyan"), "\n",
                ("Welcome back, " if returning else "Nice to meet you, ", "white"),
                (f"{name}.", "white"),
            ),
            border_style="bright_cyan",
            padding=(1, 3),
        ))
        self._write_mastery()

        # Lesson loop
        planner = Planner(self.graph, self.model)
        tutor = Tutor()

        while True:
            if planner.is_complete():
                self._write(Panel(
                    Text("All concepts mastered.", style="bold green"),
                    border_style="green", padding=(1, 3),
                ))
                self._write_mastery()
                break

            if planner.is_stuck():
                self._info("You seem stuck. Let's review some fundamentals.")
                all_ids = self.graph.get_all_concept_ids()
                concept_id = min(all_ids, key=lambda c: self.model.predict_mastery(c))
            else:
                concept_id = planner.select_next_concept()
                if concept_id is None:
                    self._write(Panel(
                        Text("All concepts mastered.", style="bold green"),
                        border_style="green", padding=(1, 3),
                    ))
                    break

            cont = await self._run_lesson(concept_id, planner, self.model, tutor)
            if not cont:
                return

            self._info("Press Enter to continue, or type 'exit' to stop.")
            choice = await self._ask(allow_empty=True)
            if choice is None:
                self._save_and_quit()
                return

    async def _run_lesson(
        self, concept_id: str, planner: Planner, model: StudentModel, tutor: Tutor,
    ) -> bool:
        """Run one lesson. Returns False if student wants to quit."""
        context = planner.build_context(concept_id)
        node = self.graph.get_node(concept_id)
        prereqs = self.graph.get_prerequisites(concept_id)
        lesson_start = time.time()

        # Lesson header
        title = node.lesson_title or node.description
        self._write_lesson_header(node.name, title, context.student_mastery)

        # Premise check
        if prereqs:
            weakest = min(prereqs, key=lambda p: model.predict_mastery(p))
            prereq_node = self.graph.get_node(weakest)

            try:
                question = await self._call(
                    tutor.check_premise, prereq_node.name, prereq_node.description
                )
            except Exception as e:
                self._error(str(e))
                model.save(self.student_path)
                return False

            self._tutor_says(question)
            answer = await self._ask()
            if answer is None:
                model.save(self.student_path)
                return False

            try:
                result = await self._call(tutor.evaluate_response, prereq_node.name, answer)
            except Exception as e:
                self._error(str(e))
                model.save(self.student_path)
                return False

            model.partial_update(weakest, result["understood"])
            if result["understood"]:
                self._info(f"Good. {result['note']}")
            else:
                self._info(result["note"])
                self._info(f"We'll keep going, but you may want to revisit {prereq_node.name} later.")

        # Pre-lesson questions
        self._tutor_says("Any questions before we get into it?")
        pre_q = await self._ask()
        if pre_q is None:
            model.save(self.student_path)
            return False

        if pre_q.lower() not in ("no", "n", "none", "nope", ""):
            try:
                text = await self._stream_tutor(
                    tutor.continue_teaching_stream, context, pre_q
                )
                tutor.record_response(text, context)
            except Exception as e:
                self._error(str(e))
                model.save(self.student_path)
                return False

        # Teaching phase (streaming)
        try:
            text = await self._stream_tutor(tutor.start_teaching_stream, context)
            tutor.record_response(text, context)
        except Exception as e:
            self._error(str(e))
            model.save(self.student_path)
            return False

        while True:
            answer = await self._ask()
            if answer is None:
                model.save(self.student_path)
                return False

            if answer.lower() in ("ready", "done", "next", "move on"):
                break

            try:
                text = await self._stream_tutor(
                    tutor.continue_teaching_stream, context, answer
                )
                tutor.record_response(text, context)
            except Exception as e:
                self._error(str(e))
                model.save(self.student_path)
                return False

        # Mid-lesson understanding eval
        try:
            mid_result = await self._call(
                tutor.evaluate_response, context.concept_name,
                "Based on the conversation so far, does this student seem to understand the concept?",
            )
            model.partial_update(concept_id, mid_result["understood"])
        except Exception:
            pass

        # Assessment (adaptive)
        try:
            question = await self._call(tutor.ask_assessment_question, context)
        except Exception as e:
            self._error(str(e))
            model.save(self.student_path)
            return False

        for _ in range(4):
            self._tutor_says(question)
            answer = await self._ask()
            if answer is None:
                model.save(self.student_path)
                return False

            try:
                result = await self._call(tutor.judge_answer, context, answer)
            except Exception as e:
                self._error(str(e))
                model.save(self.student_path)
                return False

            model.update(concept_id, result["correct"])
            self._write_assessment(result)

            confidence = result.get("confidence", "clear_pass" if result["correct"] else "clear_fail")
            if confidence != "uncertain":
                break

            try:
                question = await self._call(tutor.ask_followup_question, context)
            except Exception as e:
                self._error(str(e))
                model.save(self.student_path)
                return False

        # Lesson summary
        new_mastery = model.predict_mastery(concept_id)
        duration = time.time() - lesson_start
        model.save(self.student_path)

        next_concept = planner.select_next_concept()
        next_name = None
        next_title = None
        if next_concept and not planner.is_complete():
            next_node = self.graph.get_node(next_concept)
            next_name = next_node.name
            next_title = next_node.lesson_title or None

        self._write_lesson_summary(node.name, new_mastery, duration, next_name, next_title)
        self._write_mastery()

        return True


def _split_prose_and_code(text: str) -> list[tuple[str, str]]:
    segments = []
    pattern = re.compile(r"(```[^\n]*\n.*?```)", re.DOTALL)
    last_end = 0
    for match in pattern.finditer(text):
        if match.start() > last_end:
            segments.append(("prose", text[last_end:match.start()]))
        segments.append(("code", match.group(0)))
        last_end = match.end()
    if last_end < len(text):
        segments.append(("prose", text[last_end:]))
    return segments
