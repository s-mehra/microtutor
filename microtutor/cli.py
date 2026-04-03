"""Entry point for microtutor."""

from pathlib import Path

from microtutor.app import MicrotutorApp

DATA_DIR = Path(__file__).parent.parent / "data"
STUDENTS_DIR = Path(__file__).parent.parent / "students"


def main() -> None:
    graph_path = DATA_DIR / "neural_networks.json"
    app = MicrotutorApp(graph_path=graph_path, students_dir=STUDENTS_DIR)
    app.run()


if __name__ == "__main__":
    main()
