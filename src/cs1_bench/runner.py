"""Run one model over decision variants and write row-level predictions plus a run manifest."""

from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

from . import __version__, ui
from .cli import add_model_arguments, check_model_arguments, load_backend
from .conditions import VARIANT_FIELDS
from .core import prompt_condition, supports_system_role, validate_row

MANIFEST_SCHEMA = "cs1-bench-run-v1"
# Copied from each input row into its prediction, so the predictions file is self-contained.
CARRIED_FIELDS = (*VARIANT_FIELDS, "answer", "lang", "group_id")
# The reranker builds its own prompt, so prompt conditions do not apply to it.
PROMPT_FIELDS = ("prompt_lang", "prompt_format")


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _git_state() -> dict:
    root = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                                text=True, check=True).stdout.strip()
        changes = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                                 capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": bool(changes)}


def _package_versions() -> dict:
    """Versions of the libraries that turn text into scores; recorded rather than pinned."""
    versions = {}
    for name in ("transformers", "tokenizers", "huggingface-hub", "accelerate", "safetensors"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _environment() -> dict:
    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "machine": platform.machine(), "packages": _package_versions()}
    try:
        import torch
    except ImportError:
        return environment
    environment["torch"] = torch.__version__
    environment["torch_cuda"] = torch.version.cuda
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        environment["gpu"] = properties.name
        # Recorded because builds may lack official kernels for new architectures (e.g. sm_121).
        environment["gpu_capability"] = f"{properties.major}.{properties.minor}"
        environment["gpu_memory_gib"] = round(properties.total_memory / 2**30, 1)
    return environment


def resolve_system_roles(rows: list[dict], tokenizer) -> tuple[list[dict], bool, int]:
    """Send our system prompt as a system turn, or, if the chat template has none, at the top of the user turn.

    Any system_role already in a row is ignored: the template decides, never the input file.
    Scorers record the role actually used in each result as system_role (native or merge).
    """
    supported = supports_system_role(tokenizer)
    role = "native" if supported else "merge"
    return [{**row, "system_role": role} for row in rows], supported, 0 if supported else len(rows)


def scoring_units(rows: list[dict], mode: str) -> list[list[dict]]:
    """Split rows into units: one row each, or one group per exact state and prompt condition."""
    if mode not in {"serial", "shared"}:
        return [[row] for row in rows]
    groups: dict[str, list[dict]] = {}
    for row in rows:
        key = json.dumps([row["state"], prompt_condition(row)], ensure_ascii=False, sort_keys=True)
        groups.setdefault(key, []).append(row)
    return list(groups.values())


def _carried(row: dict, mode: str) -> dict:
    fields = [field for field in CARRIED_FIELDS if mode != "reranker" or field not in PROMPT_FIELDS]
    return {field: row[field] for field in fields if field in row}


def _score_unit(unit, mode, scorers, model, tokenizer, metadata, max_tokens, serial_scorer) -> list[dict]:
    """Score one unit; a ValueError marks its rows as failed instead of stopping the run."""
    if mode == "shared":
        try:
            results, timing = scorers.shared(model, tokenizer, unit, metadata, max_tokens)
        except ValueError as error:
            return [{**_carried(row, mode), "id": row["id"], "error": str(error)} for row in unit]
        return [{**_carried(row, mode), **result, "shared_timing": timing} for row, result in zip(unit, results)]
    records = []
    for row in unit:
        try:
            if mode == "serial":
                result = serial_scorer.score(row)
            else:
                result = getattr(scorers, mode)(model, tokenizer, row, metadata, max_tokens)
        except ValueError as error:
            result = {"id": row["id"], "error": str(error)}
        records.append({**_carried(row, mode), **result})
    return records


def _warm_up(units, mode, scorers, model, tokenizer, metadata, max_tokens, repeats: int) -> None:
    """Run the first unit untimed so one-off kernel and allocator setup does not skew latencies."""
    for _ in range(repeats):
        serial_scorer = scorers.serial(model, tokenizer, metadata, max_tokens) if mode == "serial" else None
        _score_unit(units[0], mode, scorers, model, tokenizer, metadata, max_tokens, serial_scorer)


def run(rows: list[dict], output_dir: Path, args: argparse.Namespace, input_path: Path) -> dict:
    """Score every row with one model; always leave a manifest describing what happened."""
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "running",
        "cs1_bench_version": __version__,
        "code": _git_state(),
        "command": sys.argv,
        "mode": args.mode,
        "backend": args.backend,
        "model": {"source": args.model, "revision": args.revision, "revision_source": args.revision_source},
        "input": {"path": str(input_path), "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                  "rows": len(rows)},
        "warmup_repeats": args.warmup,
        "environment": _environment(),
        "started_utc": _utc_now(),
    }
    started = time.perf_counter()
    progress = ui.Progress(len(rows), args.progress_every)
    try:
        progress.say(f"loading weights ({args.backend}, {args.mode})")
        if args.allow_remote_code:
            files = ", ".join(sorted(path.name for path in Path(args.model).glob("*.py"))) or "no local .py files"
            progress.say(f"remote code allowed: running {files}", ui.WARN)
        model, tokenizer, metadata, scorers = load_backend(args)
        manifest["model"] = {**metadata, "revision_source": args.revision_source}
        manifest["load_seconds"] = time.perf_counter() - started
        progress.say(ui.loaded(manifest), ui.OK)
        if args.mode != "reranker":
            rows, supported, fallbacks = resolve_system_roles(rows, tokenizer)
            manifest["system_role_supported"], manifest["system_role_fallbacks"] = supported, fallbacks
            if not supported:
                progress.say(f"no system role in the chat template: system prompt goes in the user turn "
                             f"({fallbacks} rows)", ui.WARN)
        manifest["conditions"] = {
            field: dict(collections.Counter(str(row.get(field, prompt_condition(row).get(field))) for row in rows))
            for field in ("prompt_lang", "prompt_format", "system_role", "rotation")
        }
        units = scoring_units(rows, args.mode)
        if args.warmup:
            progress.say(f"warm-up ({args.warmup}x, untimed)")
        _warm_up(units, args.mode, scorers, model, tokenizer, metadata, args.max_tokens, args.warmup)
        serial_scorer = scorers.serial(model, tokenizer, metadata, args.max_tokens) if args.mode == "serial" else None
        groups = f" in {len(units)} groups" if args.mode in {"serial", "shared"} else ""
        progress.say(f"scoring {ui.plural(len(rows), 'row')}{groups}")
        scoring_started = time.perf_counter()
        progress.start()
        with (output_dir / "predictions.jsonl").open("x", encoding="utf-8") as destination:
            for unit in units:
                records = _score_unit(unit, args.mode, scorers, model, tokenizer, metadata,
                                      args.max_tokens, serial_scorer)
                for record in records:
                    destination.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                destination.flush()
                progress.advance(records)
        manifest["scoring_seconds"] = time.perf_counter() - scoring_started
        manifest["rows_scored"], manifest["rows_failed"] = progress.done - progress.failed, progress.failed
        manifest["status"] = "completed"
        progress.close()
        status = ui.WARN if progress.failed else ui.OK
        progress.say(ui.scored(manifest), status)
        progress.say(ui.finished(time.perf_counter() - started, output_dir, progress.failed), status)
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        progress.close()
        progress.say(f"FAILED: {manifest['error']} (manifest written to {output_dir})", ui.FAIL)
        raise
    finally:
        manifest["finished_utc"] = _utc_now()
        manifest["wall_seconds"] = time.perf_counter() - started
        with (output_dir / "manifest.json").open("x", encoding="utf-8") as destination:
            destination.write(json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Decisions or variants JSONL (from cs1 import or cs1 variants)")
    parser.add_argument("--output", type=Path, required=True, help="New directory for predictions and manifest")
    parser.add_argument("--warmup", type=int, default=1, help="Untimed repeats of the first unit (default: 1)")
    parser.add_argument("--progress-every", type=float, default=10.0, metavar="SECONDS",
                        help="Seconds between progress lines on stderr; 0 prints only start and end (default: 10)")
    add_model_arguments(parser)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory must be new")
    if args.warmup < 0 or args.progress_every < 0:
        parser.error("--warmup and --progress-every must be nonnegative")
    check_model_arguments(parser, args)
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        parser.error("Input is empty")
    for row in rows:
        try:
            validate_row(row)
        except ValueError as error:
            parser.error(f"row {row.get('id')!r}: {error}")
    if len({row["id"] for row in rows}) != len(rows):
        parser.error("Row IDs must be unique")
    if not os.environ.get(ui.NESTED):
        ui.banner()
        ui.title(f"Starting test for {Path(args.model).name} "
                 f"({args.backend}, {args.mode}, {ui.plural(len(rows), 'row')})")
        if ui.is_terminal(sys.stderr):
            # The same run in a child process, so transformers' own output is tidied like everything else.
            code = ui.follow([sys.executable, "-m", "cs1_bench", "run", *sys.argv[1:]], sys.stderr)
            ui.report_run(args.output)
            raise SystemExit(code)
    run(rows, args.output, args, args.input)


if __name__ == "__main__":
    main()
