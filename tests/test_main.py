import pytest

from cs1_bench import __main__ as cs1


def test_help_lists_every_command(capsys):
    cs1.main([])
    out = capsys.readouterr().out
    assert all(command in out for command in ("import", "variants", "run", "score"))


def test_unknown_command_exits_with_usage(capsys):
    with pytest.raises(SystemExit) as error:
        cs1.main(["evaluate-everything"])
    assert error.value.code == 2
    assert "unknown command" in capsys.readouterr().err


def test_subcommand_help_names_the_subcommand(capsys):
    with pytest.raises(SystemExit) as error:
        cs1.main(["run", "--help"])
    assert error.value.code == 0
    assert capsys.readouterr().out.startswith("usage: cs1 run")
