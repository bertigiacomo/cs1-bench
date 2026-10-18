import io
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


def test_banner_spells_the_name_and_version():
    stream = io.StringIO()
    ui.banner("check", stream)
    lines = stream.getvalue().splitlines()
    assert len(lines) == 3 and lines[2].endswith(f"v{ui.__version__} · check")


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
