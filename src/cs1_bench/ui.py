"""Console look shared by cs1 run and cs1 check: banner, status marks, progress, and run summaries.

Colour and in-place updates happen on terminals only; files and pipes (`| tee log`) get plain lines.
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import __version__

OK, WARN, FAIL, SKIP, INFO, STEP = "✓", "!", "✗", "–", "·", "▸"
# ANSI colour codes: green, yellow, red, dim, dim, cyan.
COLORS = {OK: "32", WARN: "33", FAIL: "31", SKIP: "2", INFO: "2", STEP: "36"}
# Set on a cs1 run started by another cs1 command: it skips banner and title, and still draws its bar.
NESTED = "CS1_NESTED"


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


def banner(stream=None) -> None:
    title(f"====== CS1-BENCH v{__version__} ======", stream)


def title(text: str, stream=None) -> None:
    stream = sys.stderr if stream is None else stream
    print(heading(text, stream), file=stream, flush=True)


def plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def errors(count: int) -> str:
    return plural(count, "error")


def clock(seconds: float) -> str:
    return str(datetime.timedelta(seconds=round(seconds)))


def duration(seconds: float) -> str:
    """Short durations in seconds, longer ones as a clock: 53.2s, 0:10:44."""
    return f"{seconds:.1f}s" if seconds < 60 else clock(seconds)


# Summary lines, built from a run manifest so that cs1 run and cs1 check report the same way.
def loaded(manifest: dict) -> str:
    model, backend = manifest["model"], manifest["backend"]
    return (f"loaded {model.get('model_class') or backend} in {duration(manifest['load_seconds'])} "
            f"on {model.get('device', backend)}")


def scored(manifest: dict) -> str:
    rows, seconds = manifest["rows_scored"] + manifest["rows_failed"], manifest["scoring_seconds"]
    rate = rows / seconds if seconds else 0.0
    supported = manifest.get("system_role_supported")  # Absent for the reranker, which has its own prompt.
    role = "" if supported is None else f", system role {'native' if supported else 'merge'}"
    return (f"scored {plural(rows, 'row')} in {duration(seconds)} ({rate:.1f} rows/s), "
            f"{errors(manifest['rows_failed'])}{role}")


def finished(seconds: float, output_dir, failed: int) -> str:
    hint = " (failed rows have an 'error' field in predictions.jsonl)" if failed else ""
    return f"done in {clock(seconds)}, {output_dir}{hint}"


def report_run(output_dir, stream=None) -> None:
    """Summarise a run written by a child process; its warnings and errors are already on screen."""
    stream = sys.stderr if stream is None else stream
    manifest_path = Path(output_dir) / "manifest.json"
    if not manifest_path.is_file():
        print(line(FAIL, "run did not start: see the messages above", stream), file=stream)
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "completed":
        return  # The child already printed its ✗ FAILED line.
    failed = manifest["rows_failed"]
    status = WARN if failed else OK
    print(line(OK, loaded(manifest), stream), file=stream)
    print(line(status, scored(manifest), stream), file=stream)
    print(line(status, finished(manifest["wall_seconds"], output_dir, failed), stream), file=stream, flush=True)


def is_progress(text: str) -> bool:
    """Progress lines (our ·, ▸, ✓ and tqdm bars) are worth seeing only while running; warnings and errors stay."""
    return not text.strip() or text.startswith((f"  {INFO} ", f"  {STEP} ", f"  {OK} ")) or "%|" in text


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


def follow(command: list[str], stream, first: str = "") -> int:
    """Run a cs1 command in a child process: its progress on the status line, anything else kept on screen.

    Only lines recognised as progress are hidden, so output in a format we don't know is shown, never lost.
    Returns the child's exit code.
    """
    status = StatusLine(stream)
    if first:
        status.show(first)
    # Text mode reads tqdm's carriage-return updates as separate lines; stdout only repeats the summary.
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                               env={**os.environ, NESTED: "1"})
    try:
        for text in process.stderr:
            text = text.rstrip("\n")
            if is_progress(text):
                status.show(text)
            else:
                status.clear()
                print(text, file=stream, flush=True)
    finally:
        process.wait()  # On Ctrl-C the child stops too, and still writes its manifest.
        status.clear()
    return process.returncode


class Progress:
    """Progress of a run on stderr: a live bar on terminals and in nested runs, else one line per interval."""

    def __init__(self, total: int, every: float, clock=time.monotonic, stream=None):
        self.total, self.every, self.clock = total, every, clock
        self.stream = sys.stderr if stream is None else stream
        self.started = self.last = clock()
        self.done = self.failed = 0
        self.bar = None

    def say(self, message: str, status: str = INFO) -> None:
        text = line(status, message, self.stream)
        if self.bar is not None:
            self.bar.write(text, file=self.stream)  # Printed above the bar, which is then redrawn.
        else:
            print(text, file=self.stream, flush=True)

    def start(self) -> None:
        """Start timing; draw tqdm's bar (tqdm ships with transformers) on terminals and in nested runs."""
        self.started = self.last = self.clock()
        if is_terminal(self.stream) or os.environ.get(NESTED):
            from tqdm import tqdm

            step = paint(STEP, COLORS[STEP], self.stream)
            # ascii=False: block characters, like the marks (UTF-8 terminal assumed).
            self.bar = tqdm(total=self.total, file=self.stream, unit=" rows", dynamic_ncols=True, leave=False,
                            ascii=False, bar_format=f"  {step} {{n_fmt}}/{{total_fmt}} {{bar:24}} "
                                                    "{percentage:3.0f}%, {rate_fmt}, ETA {remaining}{desc}")
            self.bar.set_description_str(", 0 errors")

    def advance(self, records: list[dict]) -> None:
        self.done += len(records)
        self.failed += sum("error" in record for record in records)
        if self.bar is not None:
            self.bar.set_description_str(f", {errors(self.failed)}", refresh=False)
            self.bar.update(len(records))
            return
        now = self.clock()
        if self.every > 0 and now - self.last >= self.every:
            self.last = now
            self.say(self.line(now), STEP)

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()
            self.bar = None

    def line(self, now: float) -> str:
        elapsed = max(now - self.started, 1e-9)
        rate = self.done / elapsed
        remaining = (self.total - self.done) / rate if rate else 0.0
        return (f"{self.done}/{self.total} ({100 * self.done / self.total:.0f}%), {rate:.1f} rows/s, "
                f"ETA {clock(remaining)}, {errors(self.failed)}")
