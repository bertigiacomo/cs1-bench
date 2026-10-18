"""Preflight check without loading weights: environment, then each model's loader, tokenizer, and prompts."""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

from . import ui
from .core import (LETTERS, TEXT_EXTRACTED_MODEL_TYPES, auto_model_class, prompt_condition, resolve_model_source,
                   resolve_revision, supports_system_role, validate_row)
from .direct import _slot_ids, encode_prompt
from .runner import _environment, _git_state, resolve_system_roles
from .ui import FAIL, OK, SKIP, WARN

SEVERITY = {OK: 0, SKIP: 0, WARN: 1, FAIL: 2}
GENERATIVE_SUFFIXES = ("ForCausalLM", "ForConditionalGeneration")
WEIGHT_SUFFIXES = {".safetensors", ".bin", ".gguf"}


def classify_checkpoint(config: dict | None, allow_remote_code: bool = False) -> tuple[str, str]:
    """Say from the raw config.json whether load_causal_model can load the checkpoint as it is."""
    from transformers import __version__ as transformers_version
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

    if config is None:
        return SKIP, "not evaluable with this engine: no config.json (not a transformers checkpoint)"
    model_type = config.get("model_type")
    architectures = config.get("architectures") or []
    if architectures and not any(name.endswith(GENERATIVE_SUFFIXES) for name in architectures):
        return SKIP, f"not evaluable with this engine: {', '.join(architectures)} does not generate text"
    if "auto_map" in config:
        if allow_remote_code:
            return WARN, f"remote code allowed (auto_map, model_type {model_type!r}): read the model's .py files first"
        return FAIL, (f"requires remote code (auto_map in config, model_type {model_type!r}); "
                      "pass --allow-remote-code after reading its .py files")
    if model_type not in CONFIG_MAPPING_NAMES:
        return FAIL, (f"requires remote code (model_type {model_type!r} unknown to transformers "
                      f"{transformers_version}); trust_remote_code is off")
    if model_type in TEXT_EXTRACTED_MODEL_TYPES:
        return OK, f"loader supported: text model of {model_type}"
    found = auto_model_class(model_type)
    if found is None:
        return SKIP, f"not evaluable with this engine: {model_type} is not a causal language model"
    multimodal = " (multimodal, fed text only)" if found[1].endswith("ForConditionalGeneration") else ""
    return OK, f"loader supported: {found[1]} via {found[0]}{multimodal}"


def load_tokenizer(folder: Path):
    """Load the tokenizer exactly as load_causal_model does, without touching the weights."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(folder, local_files_only=True, trust_remote_code=False)


class Report:
    """Print status lines and remember the worst status seen."""

    def __init__(self, stream=None):
        self.stream = sys.stdout if stream is None else stream
        self.worst = OK

    def __call__(self, status: str, message: str) -> None:
        print(ui.line(status, message, self.stream), file=self.stream, flush=True)
        if status == SKIP:
            self.worst = SKIP
        elif SEVERITY[status] > SEVERITY[self.worst]:
            self.worst = status


def check_environment(report: Report) -> None:
    environment = _environment()
    if "torch" not in environment:
        report(FAIL, "torch is not installed (on a Mac: pip install -e '.[torch]')")
    else:
        import torch

        line = f"torch {environment['torch']} · CUDA {environment.get('torch_cuda')}"
        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if count:
            line += (f" · {count} GPU: {environment['gpu']} (capability {environment['gpu_capability']}, "
                     f"{environment['gpu_memory_gib']} GiB)")
        report(OK if count == 1 else WARN, line if count == 1 else
               f"{line} · {count} CUDA GPUs visible; cs1 run --device cuda needs exactly one")
    from transformers import __version__ as transformers_version

    report(OK, f"transformers {transformers_version} · python {environment['python']} · {environment['machine']}")
    offline = os.environ.get("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes", "on"}
    report(OK if offline else WARN, "HF_HUB_OFFLINE=1" if offline else
           "HF_HUB_OFFLINE is not set: libraries may try the network (export HF_HUB_OFFLINE=1)")
    git = _git_state()
    if git["git_dirty"]:
        report(WARN, f"code at {str(git['git_commit'])[:12]} has uncommitted changes; commit before official runs")
    else:
        report(OK, f"code at commit {str(git['git_commit'])[:12]}")


def _condition_label(row: dict) -> str:
    condition = prompt_condition(row)
    return f"{condition['prompt_lang']}.{condition['prompt_format']}.{condition['system_role']}"


def check_model(folder: Path, rows: list[dict], max_tokens: int, report: Report,
                allow_remote_code: bool = False) -> None:
    config_path = folder / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else None
    status, reason = classify_checkpoint(config, allow_remote_code)
    if status == SKIP:
        report(SKIP, reason)
        return
    try:
        revision, origin = resolve_revision(str(folder), None)
        report(OK, f"revision {revision[:12]}… ({origin})")
    except ValueError as error:
        report(FAIL, str(error))
    report(status, reason)
    if "auto_map" in config:
        # Listed, never imported: the static check does not execute remote code.
        files = sorted(f"{path.name} ({path.stat().st_size / 1024:.0f} KiB)" for path in folder.glob("*.py"))
        report(WARN if allow_remote_code else FAIL, f"code that would run: {', '.join(files) or 'none found'}")
    size = sum(path.stat().st_size for path in folder.rglob("*")
               if path.suffix in WEIGHT_SUFFIXES and ".cache" not in path.parts)
    if config.get("quantization_config") or (folder / "hf_quant_config.json").is_file():
        report(WARN, f"weights {size / 2**30:.1f} GiB, pre-quantized checkpoint: loading not verified")
    else:
        report(OK, f"weights {size / 2**30:.1f} GiB")
    try:
        tokenizer = load_tokenizer(folder)
    except Exception as error:  # Tokenizer loaders fail with many exception types.
        report(FAIL, f"tokenizer does not load: {type(error).__name__}: {str(error).splitlines()[0]}")
        return
    if not getattr(tokenizer, "chat_template", None):
        report(FAIL, "tokenizer has no chat template")
        return
    if supports_system_role(tokenizer):
        report(OK, "system role supported")
    else:
        report(WARN, "no system role: cs1 run will switch native rows to merge")
    count = max((len(row["options"]) for row in rows), default=len(LETTERS))
    try:
        _slot_ids(tokenizer, count)
        report(OK, f"letters A-{LETTERS[count - 1]} are single tokens")
    except ValueError as error:
        report(FAIL, str(error))
    if not rows:
        return
    resolved, _, _ = resolve_system_roles(rows, tokenizer)
    failures, first_error, longest = collections.Counter(), {}, 0
    for row in resolved:
        try:
            longest = max(longest, len(encode_prompt(tokenizer, row, max_tokens)[0]))
        except Exception as error:  # Chat templates raise their own exception types too.
            label = _condition_label(row)
            failures[label] += 1
            first_error.setdefault(label, f"{type(error).__name__}: {error}")
    for label in sorted(failures):
        report(FAIL, f"{failures[label]} rows fail in {label}: {first_error[label]}")
    if not failures:
        labels = {_condition_label(row) for row in resolved}
        report(OK, f"all {len(resolved)} rows encode in {len(labels)} prompt conditions · "
                   f"longest prompt {longest} tokens (limit {max_tokens})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*",
                        help="Local model folders or names under --models-dir (default: every folder there)")
    parser.add_argument("--models-dir", type=Path, default=os.environ.get("CS1_MODELS_DIR"),
                        help="Folder holding one subfolder per model (default: $CS1_MODELS_DIR)")
    parser.add_argument("--input", type=Path, help="Variants JSONL whose real prompts are test-encoded")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--allow-remote-code", action="store_true",
                        help="Treat models that ship their own code (auto_map) as runnable; the code is listed, "
                             "not executed")
    args = parser.parse_args()
    if args.models_dir is not None:
        os.environ["CS1_MODELS_DIR"] = str(args.models_dir)
    if args.models:
        folders = [Path(resolve_model_source(model)) for model in args.models]
    elif args.models_dir is not None and Path(args.models_dir).expanduser().is_dir():
        folders = sorted(path for path in Path(args.models_dir).expanduser().iterdir()
                         if path.is_dir() and not path.name.startswith("."))
    else:
        parser.error("Name the models, or pass --models-dir / set CS1_MODELS_DIR")
    missing = [str(folder) for folder in folders if not folder.is_dir()]
    if missing:
        parser.error(f"Not local model folders: {', '.join(missing)}")
    rows = []
    if args.input is not None:
        rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in rows:
            try:
                validate_row(row)
            except ValueError as error:
                parser.error(f"row {row.get('id')!r}: {error}")
    ui.banner("check", sys.stdout)
    print("\n" + ui.heading("Environment", sys.stdout))
    environment = Report()
    check_environment(environment)
    outcomes = collections.Counter()
    for folder in folders:
        print("\n" + ui.heading(folder.name, sys.stdout))
        report = Report()
        check_model(folder, rows, args.max_tokens, report, args.allow_remote_code)
        outcomes[report.worst] += 1
    counts = [(OK, "ready"), (WARN, "with warnings"), (FAIL, "unsupported"), (SKIP, "not evaluable")]
    print("\n" + ui.heading("Summary", sys.stdout) + "  " + "  ".join(
        f"{ui.paint(status, ui.COLORS[status], sys.stdout)} {outcomes[status]} {label}" for status, label in counts))
    if environment.worst == FAIL or outcomes[FAIL]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
