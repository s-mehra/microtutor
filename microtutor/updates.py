"""Check for new releases on GitHub."""

from __future__ import annotations

import json
import urllib.request

from microtutor import __version__

GITHUB_REPO = "s-mehra/microtutor"
RELEASES_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
TIMEOUT_SECONDS = 3


def check_for_update() -> str | None:
    """Check GitHub for a newer release.

    Returns the new version string if an update is available, or None.
    Fails silently on any error (network, auth, no releases, etc.).
    """
    try:
        req = urllib.request.Request(
            RELEASES_URL,
            headers={"Accept": "application/vnd.github.v3+json"},
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
            data = json.loads(resp.read())
            latest = data["tag_name"].lstrip("v")
            if latest != __version__:
                return latest
    except Exception:
        pass
    return None


def update_message(new_version: str) -> str:
    """Format the update notification message."""
    return (
        f"A new version of microtutor is available (v{new_version}). "
        f"Run: pip install --upgrade git+https://github.com/{GITHUB_REPO}.git"
    )
