# Contributing to microtutor

Contributions are welcome. Here's how to get started.

## Setup

```bash
git clone https://github.com/s-mehra/microtutor.git
cd microtutor
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running tests

```bash
pytest
```

All tests must pass before submitting a PR.

## Code style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting. Run it before submitting:

```bash
ruff check .
```

## Submitting changes

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Add tests for new functionality
4. Make sure all tests pass and ruff is clean
5. Open a pull request

## Review process

All PRs require review from the maintainer before merging. CI must pass (pytest + ruff).

## Reporting issues

Open an issue on GitHub. Include steps to reproduce and any relevant error output.
