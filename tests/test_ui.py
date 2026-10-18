import io

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
