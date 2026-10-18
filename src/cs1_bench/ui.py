"""Console look shared by cs1 run and cs1 check: a small banner, status marks, colour only on terminals."""

from __future__ import annotations

import os
import sys

from . import __version__

BANNER = (
    "█▀▀ █▀▀ ▄█    █▄▄ █▀▀ █▄ █ █▀▀ █ █",
    "█▄▄ ▄▄█  █ ▀▀ █▄█ ██▄ █ ▀█ █▄▄ █▀█",
)
OK, WARN, FAIL, SKIP, INFO, STEP = "✓", "!", "✗", "–", "·", "▸"
# ANSI colour codes: green, yellow, red, dim, dim, cyan.
COLORS = {OK: "32", WARN: "33", FAIL: "31", SKIP: "2", INFO: "2", STEP: "36"}


def is_terminal(stream) -> bool:
    """True for an interactive terminal (tmux included), false for files and pipes such as `| tee log`."""
    return bool(getattr(stream, "isatty", lambda: False)())


def paint(text: str, code: str, stream) -> str:
    """Colour text for terminals only, honouring the NO_COLOR convention."""
    return f"\033[{code}m{text}\033[0m" if is_terminal(stream) and "NO_COLOR" not in os.environ else text


def line(status: str, message: str, stream) -> str:
    return f"  {paint(status, COLORS[status], stream)} {message}"


def heading(text: str, stream) -> str:
    return paint(text, "1", stream)


def banner(command: str, stream=None) -> None:
    stream = sys.stderr if stream is None else stream
    for row in BANNER:
        print("  " + paint(row, "36;1", stream), file=stream)
    print("  " + paint(f"Classification System One Benchmark · v{__version__} · {command}", "2", stream),
          file=stream, flush=True)
