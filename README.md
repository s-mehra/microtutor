# microtutor

A knowledge-state aware AI tutor that remembers what you know, tracks what you struggle with, and sequences material accordingly.

Most AI tutors are stateless. Every conversation starts from zero. Microtutor closes the loop: **teach, assess, remember, adapt**.

## How it works

Microtutor has three layers:

- **Student Model** -- Bayesian Knowledge Tracing (BKT) maintains a mastery score (0-1) for every concept. Correct answers increase mastery. Incorrect answers reveal gaps. The math is real, not vibes.
- **Curriculum Planner** -- A graph-based planner that finds the "frontier" (concepts where prerequisites are met but mastery is low) and decides what to teach next. If you're stuck, it backtracks to reinforce prerequisites.
- **Conversational Layer** -- Claude (via the Anthropic API) receives structured context from the planner and teaches using Socratic questioning, diagrams, and worked examples. The LLM decides *how* to teach. The planner decides *what* to teach.

## Installation

Requires Python 3.11+ and an [Anthropic API key](https://console.anthropic.com/).

```bash
git clone https://github.com/your-username/microtutor.git
cd microtutor
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
export ANTHROPIC_API_KEY=sk-ant-...
microtutor
```

The app launches a terminal UI. It will ask your name, then begin teaching from the concept graph (currently: Intro to Neural Networks, 15 concepts from vectors to multi-layer perceptrons).

Your progress is saved automatically in `students/` as a JSON file. Come back anytime and pick up where you left off.

## Configuration

Set the model via environment variable (defaults to Claude Sonnet):

```bash
export MICROTUTOR_MODEL=claude-sonnet-4-6
```

## Running tests

```bash
source .venv/bin/activate
pytest
```

## Project structure

```
microtutor/
  app.py          # Textual TUI application
  cli.py          # Entry point
  graph.py        # Concept graph (NetworkX DAG)
  model.py        # Student model (BKT)
  planner.py      # Curriculum planner
  tutor.py        # Claude API conversational layer
data/
  neural_networks.json   # 15-node concept graph
students/                # Per-student progress (gitignored)
tests/                   # Unit tests (29 tests)
```

## Contributing

Contributions welcome via pull requests. Please include tests for new functionality and make sure existing tests pass.

## License

MIT
