"""Run one model over decision variants and write common-format rows plus a run manifest."""

from __future__ import annotations

import argparse
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

from . import output, ui
from .cli import add_model_arguments, check_model_arguments, load_backend
from .core import prompt_condition, supports_system_role, validate_row
from .direct import PROMPT_VERSION


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


def _score_unit(unit, mode, scorers, model, tokenizer, metadata, max_tokens, serial_scorer) -> list[dict]:
    """Score one unit and return one result per row; a ValueError marks its rows as failed instead of stopping the run."""
    if mode == "shared":
        try:
            results, _ = scorers.shared(model, tokenizer, unit, metadata, max_tokens)
        except ValueError as error:
            return [{"error": str(error)} for _ in unit]
        return results
    results = []
    for row in unit:
        try:
            if mode == "serial":
                results.append(serial_scorer.score(row))
            else:
                results.append(getattr(scorers, mode)(model, tokenizer, row, metadata, max_tokens))
        except ValueError as error:
            results.append({"error": str(error)})
    return results


def _warm_up(units, mode, scorers, model, tokenizer, metadata, max_tokens, repeats: int) -> None:
    """Run the first unit untimed so one-off kernel and allocator setup does not skew latencies."""
    for _ in range(repeats):
        serial_scorer = scorers.serial(model, tokenizer, metadata, max_tokens) if mode == "serial" else None
        _score_unit(units[0], mode, scorers, model, tokenizer, metadata, max_tokens, serial_scorer)


def model_id(args: argparse.Namespace) -> str:
    """The model ID of the thesis registry: --model-id, else the name of the model folder (or Hugging Face repo)."""
    return args.model_id or Path(args.model).name


def run_files(output_dir: Path, name: str) -> tuple[Path, Path]:
    return output_dir / f"{name}.predictions.jsonl", output_dir / f"{name}.manifest.json"


def _sorted_orders(orders: set[str]) -> list[str]:
    return sorted(orders, key=lambda label: (label[:3], int(label[3:])))  # dir2 before dir10


def new_manifest(rows: list[dict], args: argparse.Namespace, input_path: Path, name: str,
                 started: datetime.datetime) -> dict:
    """The 24 fields of the thesis manifest (STANDARD-DATI.md), then the few that only cs1-bench needs."""
    environment, code = _environment(), _git_state()
    return {
        "run_id": name,
        "status": "running",
        "manifest_origin": "written_by_program",
        "model_id": model_id(args),
        "wrapper_id": "cs1-bench",
        "prompt_id": PROMPT_VERSION,
        "python_version": environment["python"],
        "model_revision": args.revision,
        "dtype": args.dtype,
        "code_commit": code["git_commit"],
        "inputs": [{"dataset_id": input_path.stem, "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                    "rows": len(rows)}],
        "prompt_langs": sorted({prompt_condition(row)["prompt_lang"] for row in rows}),
        "prompt_formats": sorted({prompt_condition(row)["prompt_format"] for row in rows}),
        "orders": _sorted_orders({output.order_label(row.get("rotation")) for row in rows}),
        "rows_written": 0,
        "rows_failed": 0,
        "started_utc": started.isoformat(timespec="seconds"),
        "probabilities_kind": "softmax_option_logits",
        "confidence_kind": None,
        "seconds_kind": "local_inference",
        "temperature": None,
        "torch_version": environment.get("torch"),
        "transformers_version": environment["packages"]["transformers"],
        "gpu": environment.get("gpu"),
        # cs1-bench only: needed to trust, reproduce, or summarise the run.
        "code_dirty": code["git_dirty"],  # Uncommitted changes: the commit alone does not describe the code.
        "backend": args.backend,
        "mode": args.mode,
        "device": None,
        "model_class": None,
        "system_role": None,  # native, or merge when the chat template has no system turn
        "load_seconds": None,
        "scoring_seconds": None,
        "wall_seconds": None,
        "error": None,
    }


def run(rows: list[dict], output_dir: Path, args: argparse.Namespace, input_path: Path) -> dict:
    """Score every row with one model; always leave a manifest describing what happened."""
    started_at = datetime.datetime.now(datetime.timezone.utc)
    name = args.run_id or output.run_id(model_id(args), input_path.stem, started_at)
    predictions_path, manifest_path = run_files(output_dir, name)
    if predictions_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Run {name} already has files in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = new_manifest(rows, args, input_path, name, started_at)
    started = time.perf_counter()
    progress = ui.Progress(len(rows), args.progress_every)
    try:
        progress.say(f"loading weights ({args.backend}, {args.mode})")
        if args.allow_remote_code:
            files = ", ".join(sorted(path.name for path in Path(args.model).glob("*.py"))) or "no local .py files"
            progress.say(f"remote code allowed: running {files}", ui.WARN)
        model, tokenizer, metadata, scorers = load_backend(args)
        manifest.update(dtype=metadata.get("dtype", args.dtype), device=metadata.get("device"),
                        model_class=metadata.get("model_class"), load_seconds=time.perf_counter() - started)
        progress.say(ui.loaded(manifest), ui.OK)
        rows, supported, fallbacks = resolve_system_roles(rows, tokenizer)
        manifest["system_role"] = "native" if supported else "merge"
        if not supported:
            progress.say(f"no system role in the chat template: system prompt goes in the user turn "
                         f"({fallbacks} rows)", ui.WARN)
        units = scoring_units(rows, args.mode)
        if args.warmup:
            progress.say(f"warm-up ({args.warmup}x, untimed)")
        _warm_up(units, args.mode, scorers, model, tokenizer, metadata, args.max_tokens, args.warmup)
        serial_scorer = scorers.serial(model, tokenizer, metadata, args.max_tokens) if args.mode == "serial" else None
        groups = f" in {len(units)} groups" if args.mode in {"serial", "shared"} else ""
        progress.say(f"scoring {ui.plural(len(rows), 'row')}{groups}")
        scoring_started = time.perf_counter()
        progress.start()
        with predictions_path.open("x", encoding="utf-8") as destination:
            for unit in units:
                results = _score_unit(unit, args.mode, scorers, model, tokenizer, metadata,
                                      args.max_tokens, serial_scorer)
                records = [output.common_row(name, row, result) for row, result in zip(unit, results)]
                for record in records:
                    destination.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                destination.flush()
                progress.advance(records)
        manifest["scoring_seconds"] = time.perf_counter() - scoring_started
        manifest["rows_written"], manifest["rows_failed"] = progress.done, progress.failed
        manifest["status"] = "partial" if progress.failed else "completed"
        progress.close()
        status = ui.WARN if progress.failed else ui.OK
        progress.say(ui.scored(manifest), status)
        progress.say(ui.finished(time.perf_counter() - started, output_dir, progress.failed), status)
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        progress.close()
        progress.say(f"FAILED: {manifest['error']} (manifest written to {manifest_path})", ui.FAIL)
        raise
    finally:
        manifest["wall_seconds"] = time.perf_counter() - started
        with manifest_path.open("x", encoding="utf-8") as destination:
            destination.write(json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Decisions or variants JSONL (from cs1 import or cs1 variants)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Folder for the run files <run_id>.predictions.jsonl and <run_id>.manifest.json "
                             "(created if missing; existing files are never overwritten)")
    parser.add_argument("--model-id", help="Model ID for the run ID and manifest (default: the model folder name)")
    parser.add_argument("--run-id", help=argparse.SUPPRESS)  # Set by the parent when it follows the run in a child.
    parser.add_argument("--warmup", type=int, default=1, help="Untimed repeats of the first unit (default: 1)")
    parser.add_argument("--progress-every", type=float, default=10.0, metavar="SECONDS",
                        help="Seconds between progress lines on stderr; 0 prints only start and end (default: 10)")
    add_model_arguments(parser)
    args = parser.parse_args()
    if args.warmup < 0 or args.progress_every < 0:
        parser.error("--warmup and --progress-every must be nonnegative")
    check_model_arguments(parser, args)
    if args.run_id is None:
        try:
            args.run_id = output.run_id(model_id(args), args.input.stem,
                                        datetime.datetime.now(datetime.timezone.utc))
        except ValueError as error:
            parser.error(str(error))
    if any(path.exists() for path in run_files(args.output, args.run_id)):
        parser.error(f"Run {args.run_id} already has files in {args.output}")
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        parser.error("Input is empty")
    for row in rows:
        try:
            validate_row(row)
            output.order_label(row.get("rotation"))
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
            child = [*sys.argv[1:]] if "--run-id" in sys.argv else [*sys.argv[1:], "--run-id", args.run_id]
            code = ui.follow([sys.executable, "-m", "cs1_bench", "run", *child], sys.stderr)
            ui.report_run(args.output, args.run_id)
            raise SystemExit(code)
    run(rows, args.output, args, args.input)


if __name__ == "__main__":
    main()
