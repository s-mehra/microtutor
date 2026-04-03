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
from textual.widgets import Header, Input, Static

from microtutor.config import AppConfig, ConfigManager
from microtutor.course import CourseManager, CourseMeta
from microtutor.graph import ConceptGraph
from microtutor.history import LessonHistory, LessonRecord
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

    def __init__(self, config: ConfigManager) -> None:
        super().__init__()
        self.config = config
        self.app_config: AppConfig | None = None
        self.course_manager: CourseManager | None = None
        self.graph: ConceptGraph | None = None
        self.model: StudentModel | None = None
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
        if self.model and self.course_manager:
            self.course_manager.save_student(self.model)
            if self.graph:
                meta = self.course_manager.load_meta()
                self.course_manager.update_meta_stats(meta, self.model, self.graph)
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
        # Welcome banner
        self._write(Panel(
            Text.assemble(
                ("microtutor", "bold bright_cyan"), "\n",
                ("Knowledge-State Aware AI Tutor", "dim"),
            ),
            border_style="bright_cyan",
            padding=(1, 3),
        ))

        # Check for updates (non-blocking)
        try:
            from microtutor.updates import check_for_update, update_message
            new_version = await asyncio.to_thread(check_for_update)
            if new_version:
                self._write(Panel(
                    Text(update_message(new_version), style="yellow"),
                    border_style="yellow",
                    padding=(0, 2),
                ))
        except Exception:
            pass

        # Onboarding or load config
        if not self.config.exists():
            if not await self._onboard():
                return
        self.app_config = self.config.load()

        # Resolve API key
        api_key = self.config.get_api_key()
        if not api_key:
            self._error("No API key found. Set ANTHROPIC_API_KEY or re-run onboarding.")
            return

        self._info(f"Welcome back, {self.app_config.name}.")

        # Course menu
        course_dir = await self._course_menu()
        if course_dir is None:
            return

        # Load course
        self.course_manager = CourseManager(course_dir)
        meta = self.course_manager.load_meta()
        self.graph = self.course_manager.load_graph()
        self.model = self.course_manager.load_student(self.graph)

        self._info(f"Course: {meta.title}")

        # Show last lesson context on resume
        history = LessonHistory(self.course_manager.lessons_path)
        last_lesson = history.get_last_lesson()
        if last_lesson:
            self._info(
                f"Last session: {last_lesson.concept_name} "
                f"(mastery: {last_lesson.mastery_after:.0%})"
            )

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

            # Save progress after each lesson
            self.course_manager.save_student(self.model)
            self.course_manager.update_meta_stats(meta, self.model, self.graph)

            self._info("Press Enter to continue, or type 'exit' to stop.")
            choice = await self._ask(allow_empty=True)
            if choice is None:
                self._save_and_quit()
                return

    async def _onboard(self) -> bool:
        """First-time setup. Returns True if successful."""
        self._info("First time here. Let's get you set up.")

        self._info("What's your name?")
        name = await self._ask()
        if name is None:
            return False

        self._info("Enter your Anthropic API key (get one at console.anthropic.com):")
        while True:
            key = await self._ask()
            if key is None:
                return False

            self._info("Validating API key...")
            valid = await asyncio.to_thread(self.config.validate_api_key, key)
            if valid:
                break
            self._error("Invalid API key. Please try again.")

        import time as _time
        app_config = AppConfig(
            name=name,
            api_key=key,
            created_at=_time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
        )
        self.config.save(app_config)
        self._info("Setup complete.")
        return True

    async def _course_menu(self) -> Path | None:
        """Show course menu. Returns selected course directory, or None to quit."""
        courses = self.config.list_courses()

        if not courses:
            return await self._create_new_course()

        # Build course table
        table = Table(
            title="Your Courses",
            title_style="bold white",
            show_edge=False,
            padding=(0, 1),
            expand=True,
        )
        table.add_column("#", style="bright_cyan", width=3)
        table.add_column("Course", style="white", ratio=4)
        table.add_column("Progress", justify="right", ratio=2)
        table.add_column("Last Studied", justify="right", ratio=2)

        for i, course in enumerate(courses, 1):
            pct = f"{course.progress_pct:.0%}"
            progress = f"{course.concepts_mastered}/{course.total_concepts} ({pct})"
            last = _format_relative_time(course.last_session_at) if course.last_session_at else "never"
            table.add_row(str(i), course.title, progress, last)

        self._write(table)
        self._info("")
        self._info("Enter a number to continue a course, 'new' to start a new one, or 'exit' to quit.")

        while True:
            choice = await self._ask()
            if choice is None:
                return None

            if choice.lower() in ("new", "n"):
                return await self._create_new_course()

            try:
                idx = int(choice) - 1
                if 0 <= idx < len(courses):
                    return self.config.get_course_dir(courses[idx].slug)
            except ValueError:
                pass

            self._info(f"Enter a number 1-{len(courses)}, 'new', or 'exit'.")

    async def _create_new_course(self) -> Path | None:
        """Create a new course via goal conversation and graph generation."""
        from microtutor.generator import CourseGenerator, GoalBrief

        api_key = self.config.get_api_key()
        generator = CourseGenerator(api_key)

        self._info("Let's build a course for you. I'll ask a few questions about what you want to learn.")
        self._info("(This uses Claude Opus for the best results.)")
        self._info("")

        # Goal conversation (6-10 turns diagnostic)
        try:
            result = await asyncio.to_thread(generator.get_next_message, None)
        except Exception as e:
            self._error(f"Could not start course generation: {e}")
            self._info("Try the demo course instead, or check your API key has Opus access.")
            return None

        while isinstance(result, str):
            self._tutor_says(result)
            answer = await self._ask()
            if answer is None:
                return None
            try:
                result = await asyncio.to_thread(generator.get_next_message, answer)
            except Exception as e:
                self._error(f"Conversation error: {e}")
                return None

        # result is now a GoalBrief
        brief: GoalBrief = result

        self._info(f"Great. Generating your course: {brief.course_title}")
        self._info("This may take a moment...")

        # Generate graph
        try:
            graph_data = await asyncio.to_thread(generator.generate_graph, brief)
        except Exception as e:
            self._error(f"Graph generation failed: {e}")
            return None

        # Validate
        try:
            graph = generator.validate_graph(graph_data)
        except ValueError as e:
            self._error(f"Generated graph is invalid: {e}. Retrying...")
            try:
                graph_data = await asyncio.to_thread(generator.generate_graph, brief)
                graph = generator.validate_graph(graph_data)
            except Exception as e2:
                self._error(f"Retry failed: {e2}")
                return None

        # Show preview
        concept_count = len(graph.get_all_concept_ids())
        self._info(f"Generated {concept_count} concepts:")
        preview_table = Table(show_header=False, show_edge=False, padding=(0, 1), expand=True)
        preview_table.add_column("Concept", style="bright_cyan", ratio=3)
        preview_table.add_column("Topics", style="dim", ratio=5)
        for cid in graph.get_all_concept_ids():
            node = graph.get_node(cid)
            topics = ", ".join(node.key_topics[:3])
            if len(node.key_topics) > 3:
                topics += f" (+{len(node.key_topics) - 3} more)"
            preview_table.add_row(node.name, topics or "-")
        self._write(preview_table)

        self._info("Start this course? (yes/no)")
        confirm = await self._ask()
        if confirm is None or confirm.lower() in ("no", "n"):
            self._info("Course cancelled.")
            return None

        # Save everything
        course_dir = self.config.create_course_dir(brief.course_title)
        generator.save_graph(graph_data, course_dir / "graph.json")

        meta = CourseMeta(
            title=brief.course_title,
            slug=course_dir.name,
            description=brief.course_description,
            goals=brief.goals,
            background=brief.background,
            desired_depth=brief.desired_depth,
            goal_conversation=brief.conversation,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            last_session_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            total_concepts=concept_count,
        )
        CourseManager(course_dir).save_meta(meta)
        (course_dir / "lessons.jsonl").touch()

        return course_dir

    async def _run_lesson(
        self, concept_id: str, planner: Planner, model: StudentModel, tutor: Tutor,
    ) -> bool:
        """Run one lesson. Returns False if student wants to quit."""
        context = planner.build_context(concept_id)
        node = self.graph.get_node(concept_id)
        prereqs = self.graph.get_prerequisites(concept_id)
        lesson_start = time.time()

        # Inject lesson history context
        history = LessonHistory(self.course_manager.lessons_path)
        context.lesson_history_context = history.build_context_injection()

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
                self.course_manager.save_student(model)
                return False

            self._tutor_says(question)
            answer = await self._ask()
            if answer is None:
                self.course_manager.save_student(model)
                return False

            try:
                result = await self._call(tutor.evaluate_response, prereq_node.name, answer)
            except Exception as e:
                self._error(str(e))
                self.course_manager.save_student(model)
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
            self.course_manager.save_student(model)
            return False

        if pre_q.lower() not in ("no", "n", "none", "nope", ""):
            try:
                text = await self._stream_tutor(
                    tutor.continue_teaching_stream, context, pre_q
                )
                tutor.record_response(text, context)
            except Exception as e:
                self._error(str(e))
                self.course_manager.save_student(model)
                return False

        # Teaching phase (streaming)
        try:
            text = await self._stream_tutor(tutor.start_teaching_stream, context)
            tutor.record_response(text, context)
        except Exception as e:
            self._error(str(e))
            self.course_manager.save_student(model)
            return False

        while True:
            answer = await self._ask()
            if answer is None:
                self.course_manager.save_student(model)
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
                self.course_manager.save_student(model)
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
            self.course_manager.save_student(model)
            return False

        assessment_results = []
        for _ in range(4):
            self._tutor_says(question)
            answer = await self._ask()
            if answer is None:
                self.course_manager.save_student(model)
                return False

            try:
                result = await self._call(tutor.judge_answer, context, answer)
            except Exception as e:
                self._error(str(e))
                self.course_manager.save_student(model)
                return False

            model.update(concept_id, result["correct"])
            self._write_assessment(result)
            assessment_results.append(result)

            confidence = result.get("confidence", "clear_pass" if result["correct"] else "clear_fail")
            if confidence != "uncertain":
                break

            try:
                question = await self._call(tutor.ask_followup_question, context)
            except Exception as e:
                self._error(str(e))
                self.course_manager.save_student(model)
                return False

        # Lesson summary
        new_mastery = model.predict_mastery(concept_id)
        duration = time.time() - lesson_start
        self.course_manager.save_student(model)

        # Record lesson history
        lesson_summary_data = {"summary": "", "examples_used": [], "key_topics_covered": [], "conversation_digest": ""}
        try:
            lesson_summary_data = await self._call(tutor.summarize_lesson, node.name)
        except Exception:
            pass  # Non-critical, don't block

        history = LessonHistory(self.course_manager.lessons_path)
        record = LessonRecord(
            concept_id=concept_id,
            concept_name=node.name,
            started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(lesson_start)),
            ended_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            duration_seconds=duration,
            mastery_before=context.student_mastery,
            mastery_after=new_mastery,
            key_topics_covered=lesson_summary_data.get("key_topics_covered", []),
            examples_used=lesson_summary_data.get("examples_used", []),
            summary=lesson_summary_data.get("summary", ""),
            conversation_digest=lesson_summary_data.get("conversation_digest", ""),
        )
        history.append(record)

        # Save lesson as readable markdown note
        self.course_manager.save_note(
            concept_id=concept_id,
            concept_name=node.name,
            lesson_title=node.lesson_title or node.description,
            conversation=tutor.conversation_history,
            summary_data=lesson_summary_data,
            assessment_results=assessment_results,
            mastery_before=context.student_mastery,
            mastery_after=new_mastery,
            duration_seconds=duration,
        )

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


def _format_relative_time(iso_str: str) -> str:
    """Format an ISO timestamp as a relative time string."""
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        delta = now - dt
        days = delta.days
        if days == 0:
            hours = delta.seconds // 3600
            if hours == 0:
                return "just now"
            return f"{hours}h ago"
        elif days == 1:
            return "yesterday"
        elif days < 7:
            return f"{days} days ago"
        elif days < 30:
            weeks = days // 7
            return f"{weeks} week{'s' if weeks > 1 else ''} ago"
        else:
            months = days // 30
            return f"{months} month{'s' if months > 1 else ''} ago"
    except Exception:
        return iso_str


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
