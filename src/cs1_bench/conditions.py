"""Expand canonical decisions into experimental variants: prompt conditions times option rotations."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from .core import PROMPT_FORMATS, PROMPTS, validate_row
from .data import validate_decision

# none: original order only; cyclic: one shift per option; reverse: original plus reversed (2 passes).
ROTATIONS = ("none", "cyclic", "reverse")
# Fields a variant adds on top of its source decision; the runner copies them into every result.
# The system role is not a variant: the runner sends our system prompt the way each chat template accepts it.
VARIANT_FIELDS = ("base_id", "rotation", "prompt_lang", "prompt_format")


def rotate(options: list[dict], shift: int) -> list[dict]:
    """Move the first `shift` options to the end, so each option visits each position once over all shifts."""
    return options[shift:] + options[:shift]


def expand(decision: dict, langs=("en",), formats=("json",), rotations: str = "none") -> list[dict]:
    """Return every variant of one decision, in a fixed order."""
    validate_decision(decision)
    if rotations not in ROTATIONS:
        raise ValueError(f"rotations must be one of {list(ROTATIONS)}")
    options = decision["options"]
    if rotations == "cyclic":
        orders = [(shift, f"r{shift}", rotate(options, shift)) for shift in range(len(options))]
    else:
        orders = [(0, "r0", options)]
        if rotations == "reverse":
            orders.append(("reverse", "rev", options[::-1]))
    variants = []
    for lang, prompt_format, (rotation, suffix, ordered) in itertools.product(langs, formats, orders):
        variant = {
            **decision,
            "id": f"{decision['id']}#{lang}.{prompt_format}.{suffix}",
            "options": ordered,
            "base_id": decision["id"],
            "rotation": rotation,
            "prompt_lang": lang,
            "prompt_format": prompt_format,
        }
        validate_row(variant)
        variants.append(variant)
    return variants


def expand_file(source: Path, output: Path, **grid) -> int:
    """Expand every decision in a canonical JSONL file into a new JSONL file of variants."""
    decisions = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not decisions:
        raise ValueError(f"{source.name} contains no decisions")
    variants = []
    for decision in decisions:
        try:
            variants.extend(expand(decision, **grid))
        except ValueError as error:
            raise ValueError(f"decision {decision.get('id')!r}: {error}") from error
    ids = [variant["id"] for variant in variants]
    if len(ids) != len(set(ids)):
        raise ValueError("Variant IDs collide; source decision IDs must be unique")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as destination:
        for variant in variants:
            destination.write(json.dumps(variant, ensure_ascii=False, allow_nan=False) + "\n")
    return len(variants)


def _choices(value: str, allowed) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(items) - set(allowed))
    if not items or unknown or len(items) != len(set(items)):
        raise argparse.ArgumentTypeError(f"expected a comma-separated subset of {', '.join(allowed)}")
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Canonical decisions JSONL (from cs1-import)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--langs", type=lambda value: _choices(value, sorted(PROMPTS)), default=("en",),
                        help="Prompt languages, e.g. en,it (default: en)")
    parser.add_argument("--formats", type=lambda value: _choices(value, PROMPT_FORMATS), default=("json",),
                        help="Prompt formats, e.g. json,text (default: json)")
    parser.add_argument("--rotations", choices=ROTATIONS, default="none",
                        help="cyclic: one variant per option position; reverse: original plus reversed order "
                             "(default: none)")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new")
    try:
        count = expand_file(args.input, args.output, langs=args.langs, formats=args.formats,
                            rotations=args.rotations)
    except ValueError as error:
        parser.error(str(error))
    print(f"Wrote {count} variants into {args.output}")


if __name__ == "__main__":
    main()
