"""Single entry point: `cs1 <command> [options]`, also available as `python -m cs1_bench`."""

from __future__ import annotations

import importlib
import sys

# command -> (module, one-line summary); modules load lazily so `cs1 --help` stays instant.
COMMANDS = {
    "import": ("data", "Convert CSV, JSON or JSONL into canonical decisions"),
    "variants": ("conditions", "Expand decisions into prompt conditions and option rotations"),
    "check": ("check", "Preflight: environment and each model's loader, tokenizer, prompts (no weights)"),
    "run": ("runner", "Score variants with one model; write predictions and a manifest"),
    "score": ("cli", "SemIf's single-file scorer, without a manifest"),
}


def usage() -> str:
    width = max(map(len, COMMANDS))
    lines = [f"  {name:<{width}}  {summary}" for name, (_, summary) in COMMANDS.items()]
    return "usage: cs1 <command> [options]\n\ncommands:\n" + "\n".join(lines) + \
        "\n\nRun `cs1 <command> --help` for the options of one command."


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in {"-h", "--help"}:
        print(usage())
        return
    command, rest = argv[0], argv[1:]
    if command not in COMMANDS:
        print(f"cs1: unknown command {command!r}\n\n{usage()}", file=sys.stderr)
        raise SystemExit(2)
    module = importlib.import_module(f"cs1_bench.{COMMANDS[command][0]}")
    # Each command parses sys.argv itself; its help then reads "usage: cs1 <command> ...".
    sys.argv = [f"cs1 {command}", *rest]
    module.main()


if __name__ == "__main__":
    main()
