# TODOS

## V1 Remaining

### A/B Comparison Mode
**What:** CLI flag `--mode baseline` that skips BKT and prompts Claude directly for the same topic, to validate the thesis that structured BKT tutoring beats stateless prompting.
**Why:** Until you can feel the difference, the architecture is theoretical. This is the validation experiment.
**Context:** Build after the core REPL loop works. ~30 min effort. Compare session quality side-by-side.
**Depends on:** Working REPL loop.

### BKT Parameter Fitting Pipeline
**What:** Once observation logs exist (concept_id, correct/incorrect, timestamp), fit per-concept BKT parameters using maximum likelihood estimation.
**Why:** Literature defaults produce reasonable curves, but real data produces dramatically better mastery estimates.
**Context:** Needs ~50+ observations per concept. Requires scipy or similar for MLE.
**Depends on:** Observation logging (included in V1). Enough real usage data.

---

## V2 — Open Source Release

### Multi-Course Architecture
**What:** Restructure storage so one user can have many courses running simultaneously. On boot, the app asks: start a new course or continue an existing one. If continuing, show a list of all active courses with progress.
**Why:** A tutor that only teaches one hard-coded topic isn't a product. People want to learn multiple things.
**Context:** Local file-based storage in a `~/.microtutor/` directory (or wherever the app runs). Each course gets its own subdirectory with the knowledge graph, student model, and lesson history. Directory layout:
```
~/.microtutor/
  config.json              # name, API key
  courses/
    neural-networks/
      graph.json            # concept DAG
      student.json          # mastery state
      lessons.jsonl         # lesson history
    organic-chemistry/
      ...
```
**Depends on:** V1 complete.

### Lesson History & Context Memory
**What:** Save a record of every lesson: what was taught, examples used, questions asked, student responses, assessment results. Use this to: (a) reference back to previous examples in later lessons, (b) avoid repeating the same content across successive lessons, (c) make session resumes feel natural ("Last time we covered X and you were working on Y").
**Why:** Without history, every session starts cold. The BKT knows *what* you know but not *how* you learned it. History lets the tutor say "remember when we used the coffee shop directions example for vectors?" instead of re-explaining from scratch.
**Context:** Store as JSONL per course (one line per lesson). Inject a compressed summary of recent lessons into the LLM system prompt. For long histories, consider a vector store for semantic retrieval of relevant past examples, but start with recency-based truncation (last 5 lessons in full, older ones as one-line summaries).
**Depends on:** Multi-course architecture.

### Goal-Based Course Generation
**What:** When starting a new course, run a back-and-forth conversation to understand the student's goals, background, and desired depth. Use Opus 4.6 to generate the knowledge graph based on that conversation. Not just "teach me ML" but "I'm a data analyst who wants to understand enough ML to evaluate vendor models. I know basic stats but no linear algebra."
**Why:** A generic "intro to X" graph doesn't serve everyone. The power of an AI tutor is personalization from the start. The goal conversation shapes the graph: which concepts to include, how deep to go, what prerequisites to assume.
**Context:** Use Opus 4.6 for both the goal conversation and graph generation (quality matters here more than speed). The conversation produces a structured brief, the brief feeds into graph generation, the generated graph validates through the existing ConceptGraph.from_json pipeline (cycle detection, prerequisite validation). Save the goals alongside the graph so future sessions can reference them.
**Depends on:** Multi-course architecture, LLM graph generation.

### Sub-Topic Granularity in Knowledge Graphs
**What:** Each concept node in the knowledge graph should have a set of key topics that need to be covered within that concept, informed by the student's goals. For example, a "Vectors" node should specify sub-topics like: what is a vector, dot product, scalar multiplication, vector addition, geometric interpretation. These sub-topics guide the lesson content and assessment scope.
**Why:** A node like "Vectors" is too broad on its own. The tutor needs to know what specific aspects to cover. Without sub-topics, the tutor might teach one analogy about vectors and move on, missing scalar multiplication entirely. Sub-topics also enable more granular mastery tracking in the future.
**Context:** Add a `key_topics: list[str]` field to concept nodes. During graph generation (Opus 4.6), generate key topics per node based on the student's goals and desired depth. The teaching prompt should reference these topics so the tutor knows what ground to cover before moving to assessment. The assessment should verify understanding of the key topics, not just the concept name.
**Depends on:** Goal-based course generation.

### Onboarding Flow
**What:** First-time setup: collect name, Anthropic API key, store in config. On subsequent launches, load config automatically. Then present the course menu (new or resume).
**Why:** Users need their own API key to use the app. The onboarding should be frictionless — collect what's needed, validate the key works, and get into learning.
**Context:** Store config in `~/.microtutor/config.json`. Validate the API key by making a lightweight API call (e.g., list models). If invalid, prompt to re-enter. Environment variable `ANTHROPIC_API_KEY` should override the saved key for CI/advanced users.
**Depends on:** Multi-course architecture.

### Knowledge Decay / Spaced Repetition
**What:** Mastery scores decay over time so the system prompts review of concepts the student hasn't practiced recently.
**Why:** Without decay, a student who mastered vectors three weeks ago still shows 90% mastery even though they've likely forgotten some of it. Real tutors know to circle back.
**Context:** Options: modified BKT with a forget parameter, or a separate exponential decay layer with a configurable half-life applied on session load. Ties into review scheduling in the planner.
**Depends on:** Timestamps on mastery updates (already logged in observation log).

### Full TUI with prompt_toolkit
**What:** Replace the current print/input CLI with a proper terminal application using prompt_toolkit. Fixed input area at the bottom, scrollable output area above, proper keystroke isolation.
**Why:** Right now we're fighting the terminal with ANSI escape hacks. prompt_toolkit gives us an event loop, layout containers, and a real input widget. Works alongside Rich.
**Context:** prompt_toolkit powers Python's REPL and IPython. One dependency, not a framework rewrite. The input area is a real widget that captures keystrokes independently from output rendering.
**Depends on:** V1 feature-complete and stable.

### Open Source Setup
**What:** Prepare the repo for public open-source release. Soham is the sole maintainer and only person with publish authority. Everyone can contribute via PRs.
**Why:** The project is going open source. Need clear governance, contribution guidelines, and a license.
**Context:**
- LICENSE: MIT or Apache 2.0 (permissive, standard for CLI tools)
- CONTRIBUTING.md: PR process, code style, test requirements, issue templates
- Branch protection on main: require PR reviews (from Soham), require CI passing
- GitHub Actions: run pytest on PRs, lint with ruff
- PyPI publishing: GitHub Actions workflow triggered by Soham's release tags only
- CODEOWNERS: Soham as sole owner of all paths
- README: installation, usage, architecture overview, contributing link
**Depends on:** V2 features stable enough for public use.
