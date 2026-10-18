"""Console look shared by cs1 run and cs1 check: a small banner, status marks, colour only on terminals."""

from __future__ import annotations

import datetime
import os
import shutil
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


def duration(seconds: float) -> str:
    """Short durations in seconds, longer ones as a clock: 53.2s, 0:10:44."""
    return f"{seconds:.1f}s" if seconds < 60 else str(datetime.timedelta(seconds=round(seconds)))


def is_progress(line: str) -> bool:
    """Progress lines (our ·, ▸, ✓ and tqdm bars) are worth seeing only while running; warnings and errors stay."""
    return not line.strip() or line.startswith((f"  {INFO} ", f"  {STEP} ", f"  {OK} ")) or "%|" in line


class StatusLine:
    """One line rewritten in place at the bottom of a terminal (apt style); silent outside terminals."""

    def __init__(self, stream):
        self.stream, self.active = stream, is_terminal(stream)

    def show(self, text: str) -> None:
        if self.active:  # Cut to the terminal width so it never wraps onto a second line.
            width = shutil.get_terminal_size().columns - 1
            self.stream.write("\r\033[2K" + text[:width])
            self.stream.flush()

    def clear(self) -> None:
        if self.active:
            self.stream.write("\r\033[2K")
            self.stream.flush()
