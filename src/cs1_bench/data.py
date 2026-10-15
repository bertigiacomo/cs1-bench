"""Import decisions from CSV, JSON, or JSONL into canonical CS1-Bench JSONL."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

from .core import validate_row

FORMATS = ("csv", "json", "jsonl")
OPTION_PREFIX = "option_"
CANONICAL = ("id", "state", "question", "options", "answer", "lang", "group_id")


def validate_decision(row: dict) -> None:
    """Check a scoring row plus its optional gold answer and data language."""
    validate_row(row)
    if "answer" in row and row["answer"] not in {option["id"] for option in row["options"]}:
        raise ValueError(f"answer {row['answer']!r} is not one of the option IDs")
    if "lang" in row and not (isinstance(row["lang"], str) and row["lang"]):
        raise ValueError("lang must be a nonempty string")


def read_records(path: Path, fmt: str | None = None, delimiter: str | None = None) -> list[dict]:
    """Read raw records; CSV delimiter is sniffed among comma, semicolon, and tab."""
    fmt = fmt or path.suffix.lstrip(".").lower()
    if fmt not in FORMATS:
        raise ValueError(f"Unsupported format {fmt!r}; use one of {', '.join(FORMATS)}")
    text = path.read_text(encoding="utf-8-sig")
    if fmt == "jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if fmt == "json":
        records = json.loads(text)
        if not isinstance(records, list):
            raise ValueError("JSON input must be an array of decisions")
        return records
    if delimiter is None:
        try:
            delimiter = csv.Sniffer().sniff(text[:8192], delimiters=",;\t").delimiter
        except csv.Error:
            delimiter = ","
    return list(csv.DictReader(io.StringIO(text, newline=""), delimiter=delimiter))


def _options(record: dict) -> list[dict]:
    options = record.get("options")
    if options in (None, ""):
        # Wide CSV layout: one column per option, named option_<id>; empty cells are absent options.
        return [
            {"id": key[len(OPTION_PREFIX):], "description": value}
            for key, value in record.items()
            if key.startswith(OPTION_PREFIX) and value not in (None, "")
        ]
    if isinstance(options, str):
        options = json.loads(options)
    if not isinstance(options, list):
        raise ValueError("options must be a list or a JSON-encoded list")
    return [{"id": option, "description": option} if isinstance(option, str) else option for option in options]


def normalize(record: dict, mapping: dict[str, str] | None = None, lang: str | None = None) -> dict:
    """Convert one raw record to the canonical decision layout."""
    record = dict(record)
    for target, source in (mapping or {}).items():
        if source not in record:
            raise ValueError(f"mapped column {source!r} is missing")
        record[target] = record.pop(source)
    row = {"state": record.get("state"), "question": record.get("question"), "options": _options(record)}
    # Content-derived IDs stay stable when rows are reordered or other rows are added.
    row["id"] = str(record["id"]) if record.get("id") not in (None, "") else "d-" + hashlib.sha256(
        json.dumps(row, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()[:12]
    for key in ("answer", "group_id"):
        if record.get(key) not in (None, ""):
            row[key] = str(record[key])
    if lang or record.get("lang"):
        row["lang"] = lang or record["lang"]
    extra = {
        key: value for key, value in record.items()
        if key not in CANONICAL and key != "meta" and not key.startswith(OPTION_PREFIX) and value not in (None, "")
    }
    meta = {**record.get("meta", {}), **extra} if isinstance(record.get("meta"), dict) else extra
    if meta:
        row["meta"] = meta
    validate_decision(row)
    return {key: row[key] for key in (*CANONICAL, "meta") if key in row}


def import_file(source: Path, output: Path, fmt: str | None = None, delimiter: str | None = None,
                mapping: dict[str, str] | None = None, lang: str | None = None) -> int:
    """Normalize every record, check ID uniqueness, and write a new JSONL file."""
    rows = []
    for index, record in enumerate(read_records(source, fmt, delimiter), start=1):
        try:
            rows.append(normalize(record, mapping, lang))
        except (ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"{source.name} record {index}: {error}") from error
    if not rows:
        raise ValueError(f"{source.name} contains no decisions")
    ids = [row["id"] for row in rows]
    duplicates = sorted({key for key in ids if ids.count(key) > 1})
    if duplicates:
        raise ValueError(f"Duplicate decision IDs: {duplicates[:5]}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=FORMATS, help="Default: inferred from the input extension")
    parser.add_argument("--delimiter", help="CSV delimiter (default: sniffed among , ; and tab)")
    parser.add_argument("--map", action="append", default=[], metavar="FIELD=COLUMN",
                        help="Read a canonical field from a differently named column, e.g. state=evidence")
    parser.add_argument("--lang", help="Data language tag for every row, e.g. en or it")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new")
    mapping = {}
    for item in args.map:
        target, separator, source = item.partition("=")
        if not separator or target not in CANONICAL or not source:
            parser.error(f"--map expects FIELD=COLUMN with FIELD in {', '.join(CANONICAL)}")
        mapping[target] = source
    try:
        count = import_file(args.input, args.output, args.format, args.delimiter, mapping, args.lang)
    except ValueError as error:
        parser.error(str(error))
    print(f"Imported {count} decisions into {args.output}")


if __name__ == "__main__":
    main()
