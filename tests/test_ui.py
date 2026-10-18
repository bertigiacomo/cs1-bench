import io
import json
from types import SimpleNamespace

import pytest

from cs1_bench import ui


class Terminal(io.StringIO):
    def isatty(self):
        return True


def test_colour_only_on_terminals_and_never_with_no_color(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert ui.line(ui.OK, "loaded", io.StringIO()) == "  ✓ loaded"  # Files and pipes stay plain.
    assert ui.line(ui.OK, "loaded", Terminal()) == "  \033[32m✓\033[0m loaded"
    monkeypatch.setenv("NO_COLOR", "1")
    assert ui.line(ui.OK, "loaded", Terminal()) == "  ✓ loaded"


def test_banner_is_one_line_with_the_version():
    stream = io.StringIO()
    ui.banner(stream)
    assert stream.getvalue() == f"====== CS1-BENCH v{ui.__version__} ======\n"


MANIFEST = {"status": "completed", "backend": "torch", "model": {"model_class": "Mistral3ForConditionalGeneration",
            "device": "cuda:0"}, "load_seconds": 53.2, "scoring_seconds": 0.9, "rows_scored": 3, "rows_failed": 1,
            "wall_seconds": 54.4, "system_role_supported": True}


def test_summary_lines_come_from_the_manifest():
    assert ui.loaded(MANIFEST) == "loaded Mistral3ForConditionalGeneration in 53.2s on cuda:0"
    assert ui.scored(MANIFEST) == "scored 4 rows in 0.9s (4.4 rows/s), 1 error, system role native"
    assert ui.scored({**MANIFEST, "system_role_supported": None}).endswith("1 error")  # Reranker: no role.
    assert ui.finished(54.4, "runs/x", 1) == "done in 0:00:54, runs/x (failed rows have an 'error' field in predictions.jsonl)"


def test_report_run_summarises_a_finished_child_run(tmp_path):
    stream = io.StringIO()
    ui.report_run(tmp_path, stream)
    assert "✗ run did not start" in stream.getvalue()
    (tmp_path / "manifest.json").write_text(json.dumps(MANIFEST))
    stream = io.StringIO()
    ui.report_run(tmp_path, stream)
    assert stream.getvalue().splitlines() == [
        "  ✓ " + ui.loaded(MANIFEST), "  ! " + ui.scored(MANIFEST), "  ! " + ui.finished(54.4, tmp_path, 1)]


def test_follow_hides_progress_keeps_the_rest_and_returns_the_exit_code(monkeypatch):
    lines = ["  · loading weights (torch, direct)\n", "Loading weights:  50%|█████     | 1/2\n",
             "[transformers] a warning\n", "  ✗ FAILED: boom\n"]
    started = []

    def popen(command, **kwargs):
        started.append(kwargs["env"]["CS1_NESTED"])
        return SimpleNamespace(stderr=iter(lines), wait=lambda: 1, returncode=1)

    monkeypatch.setattr(ui.subprocess, "Popen", popen)
    stream = io.StringIO()  # Not a terminal: progress is dropped, kept lines are printed.
    assert ui.follow(["cs1", "run"], stream) == 1
    assert stream.getvalue() == "[transformers] a warning\n  ✗ FAILED: boom\n" and started == ["1"]


@pytest.mark.parametrize("line,progress", [
    ("  · loading weights (torch, direct)", True),
    ("  ▸ 2/4 (50%), 3.0 rows/s, ETA 0:00:01, 0 errors", True),
    ("  ✓ loaded in 53.2s on cuda:0", True),
    ("Loading weights:  45%|████▌     | 206/458 [00:22<00:27,  9.22it/s]", True),
    ("", True),
    ("[transformers] The tokenizer you are loading … incorrect regex pattern", False),
    ("  ! remote code allowed: running modeling_spark.py", False),
    ("  ✗ FAILED: RuntimeError: Checkpoint did not load completely", False),
])
def test_progress_lines_are_temporary_and_warnings_stay(line, progress):
    assert ui.is_progress(line) is progress


def test_status_line_rewrites_in_place_without_wrapping(monkeypatch):
    monkeypatch.setattr(ui.shutil, "get_terminal_size", lambda: SimpleNamespace(columns=11))
    stream = Terminal()
    status = ui.StatusLine(stream)
    status.show("0123456789ABCDEF")
    status.clear()
    assert stream.getvalue() == "\r\033[2K0123456789\r\033[2K"
    plain = io.StringIO()
    ui.StatusLine(plain).show("never shown in logs")
    assert plain.getvalue() == ""


def test_durations_are_seconds_then_a_clock():
    assert (ui.duration(53.24), ui.duration(644)) == ("53.2s", "0:10:44")
