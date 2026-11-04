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
    backend = manifest["backend"]
    return (f"loaded {manifest.get('model_class') or backend} in {duration(manifest['load_seconds'])} "
            f"on {manifest.get('device') or backend}")


def scored(manifest: dict) -> str:
    rows, seconds = manifest["rows_written"], manifest["scoring_seconds"]
    rate = rows / seconds if seconds else 0.0
    role = f", system role {manifest['system_role']}" if manifest.get("system_role") else ""
    return (f"scored {plural(rows, 'row')} in {duration(seconds)} ({rate:.1f} rows/s), "
            f"{errors(manifest['rows_failed'])}{role}")


def finished(seconds: float, output_dir, failed: int) -> str:
    hint = " (failed rows have an 'error' field)" if failed else ""
    return f"done in {clock(seconds)}, {output_dir}{hint}"


def report_run(output_dir, run_id: str, stream=None) -> None:
    """Summarise a run written by a child process; its warnings and errors are already on screen."""
    stream = sys.stderr if stream is None else stream
    manifest_path = Path(output_dir) / f"{run_id}.manifest.json"
    if not manifest_path.is_file():
        print(line(FAIL, "run did not start: see the messages above", stream), file=stream)
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] == "failed":
        return  # The child already printed its ✗ FAILED line.
    failed = manifest["rows_failed"]
    status = WARN if failed else OK
    print(line(OK, loaded(manifest), stream), file=stream)
    print(line(status, scored(manifest), stream), file=stream)
    print(line(status, finished(manifest["wall_seconds"], output_dir, failed), stream), file=stream, flush=True)


def report_completion(output_dir, run_id: str, stream=None) -> None:
    """Print completion box with exact paths of predictions and manifest."""
    stream = sys.stderr if stream is None else stream
    out_dir = Path(output_dir).resolve()
    pred_path = out_dir / f"{run_id}.predictions.jsonl"
    manifest_path = out_dir / f"{run_id}.manifest.json"
    print(f"\n[OK] Run completata con successo:", file=stream)
    print(f"     Predizioni: {pred_path}", file=stream)
    print(f"     Manifest:   {manifest_path}", file=stream, flush=True)


def is_progress(text: str) -> bool:
    """Progress lines (our ·, ▸, ✓ and tqdm bars) are worth seeing only while running; warnings and errors stay."""
    return (not text.strip() or text.startswith((f"  {INFO} ", f"  {STEP} ", f"  {OK} "))
            or "%|" in text or " | Accuratezza: " in text)


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
        self.correct = 0
        self.total_latency = 0.0
        self.bar = None

    def say(self, message: str, status: str = INFO) -> None:
        text = line(status, message, self.stream)
        if self.bar is not None:
            self.bar.write(text, file=self.stream)  # Printed above the bar, which is then redrawn.
        else:
            print(text, file=self.stream, flush=True)

    def start(self) -> None:
        """Start timing; draw progress bar on terminals and in nested runs."""
        self.started = self.last = self.clock()
        if is_terminal(self.stream) or os.environ.get(NESTED):
            from tqdm import tqdm

            self.bar = tqdm(total=self.total, file=self.stream, leave=False, bar_format="{desc}")
            init_desc = (
                f"[0/{self.total}]   0.0% | Accuratezza:   0.0% (0/0) | "
                f"Latenza media: 0.0ms | Errori: 0"
            )
            self.bar.set_description_str(init_desc, refresh=False)

    def advance(self, records: list[dict]) -> None:
        for record in records:
            self.done += 1
            if "error" in record:
                self.failed += 1
            else:
                pred = record.get("prediction")
                ans = record.get("answer")
                if pred is not None and pred == ans:
                    self.correct += 1
                if "seconds" in record and record["seconds"] is not None:
                    self.total_latency += float(record["seconds"])
                elif "forward_seconds" in record and record["forward_seconds"] is not None:
                    self.total_latency += float(record["forward_seconds"])

        evaluated = self.done - self.failed
        acc = (self.correct / evaluated * 100.0) if evaluated > 0 else 0.0
        pct = (self.done / self.total * 100.0) if self.total > 0 else 0.0
        if self.total_latency > 0 and evaluated > 0:
            avg_ms = (self.total_latency / evaluated) * 1000.0
        else:
            elapsed = max(self.clock() - self.started, 1e-9)
            avg_ms = (elapsed / evaluated * 1000.0) if evaluated > 0 else 0.0

        desc = (
            f"[{self.done}/{self.total}] {pct:5.1f}% | "
            f"Accuratezza: {acc:5.1f}% ({self.correct}/{evaluated}) | "
            f"Latenza media: {avg_ms:0.1f}ms | Errori: {self.failed}"
        )

        if self.bar is not None:
            self.bar.set_description_str(desc, refresh=False)
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
