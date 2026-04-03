"""Entry point for microtutor."""

from microtutor.app import MicrotutorApp
from microtutor.config import ConfigManager


def main() -> None:
    config = ConfigManager()
    app = MicrotutorApp(config=config)
    app.run()


if __name__ == "__main__":
    main()
