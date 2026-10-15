"""Shared input validation, prompts, model loading, and numeric helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path

COMMIT = re.compile(r"[0-9a-f]{40}")
# Multimodal checkpoints whose text model load_causal_model extracts and loads on its own.
TEXT_EXTRACTED_MODEL_TYPES = {"qwen3_5", "qwen3_5_text"}
LETTERS = "ABCDEFGHIJKLMNOP"
# Natural, parallel wording per language; option letters stay A-P in every language.
# "evidence" names the JSON key and the text-format tag; "closing" ends both formats,
# right before the position where the letter logits are read.
PROMPTS = {
    "en": {
        "system": (
            "Evaluate the given information against the stated criterion and choose exactly one of the listed "
            "options. Reply only with the corresponding uppercase letter, without adding explanations or reasoning."
        ),
        "evidence": "information", "criterion": "criterion", "options": "options",
        "letter": "letter", "description": "description",
        "criterion_label": "Criterion", "options_label": "Options",
        "closing": "Answer with the letter of the best option.",
    },
    "it": {
        "system": (
            "Valuta gli elementi forniti in base al criterio indicato e scegli una sola opzione tra quelle elencate. "
            "Rispondi esclusivamente con la lettera maiuscola corrispondente, senza aggiungere spiegazioni o ragionamenti."
        ),
        "evidence": "elementi", "criterion": "criterio", "options": "opzioni",
        "letter": "lettera", "description": "descrizione",
        "criterion_label": "Criterio", "options_label": "Opzioni",
        "closing": "Rispondi con la lettera dell'opzione migliore.",
    },
}
# json: SemIf's single JSON object; text: tagged evidence plus a plain "A. ..." multiple-choice list.
PROMPT_FORMATS = ("json", "text")
# native: a separate system turn; merge: system text prepended to the user turn.
SYSTEM_ROLES = ("native", "merge")


def prompt_condition(row: dict) -> dict:
    """Return the prompt condition of a row; absent fields mean English, JSON layout, and a system turn."""
    return {
        "prompt_lang": row.get("prompt_lang", "en"),
        "prompt_format": row.get("prompt_format", "json"),
        "system_role": row.get("system_role", "native"),
    }


def validate_row(row: dict) -> None:
    required = {"id", "state", "question", "options"}
    if not required <= row.keys():
        raise ValueError(f"Row is missing fields: {sorted(required - row.keys())}")
    if not all(isinstance(row[key], str) and row[key] for key in ("id", "question")):
        raise ValueError("id and question must be nonempty strings")
    state = row["state"]
    if not isinstance(state, (str, dict, list)) or not state:
        raise ValueError("state must be a nonempty string, object, or array")
    try:
        json.dumps(state, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("state must be finite JSON-compatible data") from error
    options = row["options"]
    if not isinstance(options, list) or not 2 <= len(options) <= len(LETTERS):
        raise ValueError("options must contain 2-16 entries")
    ids = []
    for option in options:
        if not isinstance(option, dict) or not isinstance(option.get("id"), str) or not isinstance(option.get("description"), str):
            raise ValueError("Each option needs string id and description fields")
        ids.append(option["id"])
    if len(ids) != len(set(ids)):
        raise ValueError("Option IDs must be unique")
    condition = prompt_condition(row)
    if condition["prompt_lang"] not in PROMPTS:
        raise ValueError(f"prompt_lang must be one of {sorted(PROMPTS)}")
    if condition["prompt_format"] not in PROMPT_FORMATS:
        raise ValueError(f"prompt_format must be one of {list(PROMPT_FORMATS)}")
    if condition["system_role"] not in SYSTEM_ROLES:
        raise ValueError(f"system_role must be one of {list(SYSTEM_ROLES)}")


def _evidence_block(state, prompt_format: str, text: dict) -> str:
    """Render the state-only start of the user content, shared by every question about that state."""
    if prompt_format == "json":
        return json.dumps({text["evidence"]: state}, ensure_ascii=False)[:-1]
    tag = text["evidence"]
    if isinstance(state, str) and f"</{tag}>" not in state.lower():
        body = state
    else:
        # Objects become JSON; escaping "</" keeps the state from closing the tag early.
        body = json.dumps(state, ensure_ascii=False).replace("</", "<\\/")
    return f"<{tag}>\n{body}\n</{tag}>"


def _user_content(row: dict, condition: dict, text: dict) -> str:
    descriptions = [option["description"] for option in row["options"]]
    if condition["prompt_format"] == "json":
        body = json.dumps({
            text["evidence"]: row["state"],
            text["criterion"]: row["question"],
            text["options"]: [
                {text["letter"]: LETTERS[index], text["description"]: description}
                for index, description in enumerate(descriptions)
            ],
        }, ensure_ascii=False)
    else:
        choices = "\n".join(f"{LETTERS[index]}. {description}" for index, description in enumerate(descriptions))
        body = (
            _evidence_block(row["state"], "text", text)
            + f"\n\n{text['criterion_label']}: {row['question']}\n\n{text['options_label']}:\n{choices}"
        )
    return body + "\n\n" + text["closing"]


def direct_messages(row: dict) -> list[dict]:
    validate_row(row)
    condition = prompt_condition(row)
    text = PROMPTS[condition["prompt_lang"]]
    content = _user_content(row, condition, text)
    if condition["system_role"] == "merge":
        return [{"role": "user", "content": text["system"] + "\n\n" + content}]
    return [
        {"role": "system", "content": text["system"]},
        {"role": "user", "content": content},
    ]


def evidence_prefix(row: dict) -> str:
    """Return the start of the user turn that depends only on the state and the prompt condition."""
    condition = prompt_condition(row)
    text = PROMPTS[condition["prompt_lang"]]
    evidence = _evidence_block(row["state"], condition["prompt_format"], text)
    return text["system"] + "\n\n" + evidence if condition["system_role"] == "merge" else evidence


def supports_system_role(tokenizer) -> bool:
    """Check that the chat template accepts a system turn and keeps its text."""
    marker = "cs1-system-role-probe"
    try:
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": marker}, {"role": "user", "content": "probe"}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except Exception:  # Templates reject system turns with assorted exception types.
        return False
    return marker in prompt


def softmax(values: list[float]) -> list[float]:
    if len(values) < 2 or any(not math.isfinite(value) for value in values):
        raise ValueError("Need at least two finite scores")
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = sum(weights)
    return [weight / total for weight in weights]


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def resolve_device(device: str = "auto"):
    import torch

    if device not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("Device must be auto, cuda, mps, or cpu")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps"
    if device == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("Expose exactly one CUDA GPU, for example with CUDA_VISIBLE_DEVICES")
        return torch.device("cuda:0")
    if device == "cpu":
        return torch.device("cpu")
    if not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable; use an Apple Silicon Mac with an MPS-enabled PyTorch build")
    return torch.device("mps")


def synchronize(device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def resolve_model_source(model: str) -> str:
    """Turn a bare model name into a folder under CS1_MODELS_DIR when one exists; otherwise keep it."""
    path = Path(model).expanduser()
    if path.exists():
        return str(path)
    root = os.environ.get("CS1_MODELS_DIR")
    candidate = Path(root).expanduser() / model if root else None
    return str(candidate) if candidate is not None and candidate.exists() else model


def local_revision(folder: Path) -> tuple[str, str] | None:
    """Read the commit a local model was downloaded at: hf download metadata first, then a REVISION file."""
    commits = set()
    for metadata in (folder / ".cache" / "huggingface" / "download").rglob("*.metadata"):
        lines = metadata.read_text(encoding="utf-8").splitlines()
        if lines:
            commits.add(lines[0].strip())  # Line 1 is the commit hash; line 2 is the file's etag.
    if len(commits) > 1:
        raise ValueError(f"{folder}: download metadata disagree on the revision: {sorted(commits)}")
    if commits:
        commit, origin = commits.pop(), "hf-download-metadata"
    elif (folder / "REVISION").is_file():
        commit, origin = (folder / "REVISION").read_text(encoding="utf-8").strip(), "revision-file"
    else:
        return None
    if not COMMIT.fullmatch(commit):
        raise ValueError(f"{folder}: revision {commit!r} from {origin} is not a 40-character commit hash")
    return commit, origin


def resolve_revision(source: str, revision: str | None) -> tuple[str, str]:
    """Return the revision to record and where it came from: argument, download metadata, or REVISION file."""
    if revision:
        return revision, "argument"
    if Path(source).is_dir():
        found = local_revision(Path(source))
        if found is None:
            raise ValueError(f"No revision found in {source}: pass --revision, or write the commit hash "
                             "into a REVISION file inside the model folder")
        return found
    raise ValueError("Remote models require --revision with a pinned 40-character commit hash")


def load_causal_model(source: str, revision: str, device: str = "auto", dtype: str = "bfloat16"):
    """Load one pinned causal model on the resolved Torch device."""
    import torch
    import transformers

    local = Path(source).exists()
    if not local and not COMMIT.fullmatch(revision or ""):
        raise ValueError("Remote models require a pinned 40-character commit revision")
    if local and not revision:
        raise ValueError("Local models require an explicit manifest/revision string")
    target = resolve_device(device)
    if dtype not in {"bfloat16", "float16", "float32"}:
        raise ValueError("Dtype must be bfloat16, float16, or float32")
    common = {"revision": None if local else revision, "local_files_only": local, "trust_remote_code": False}
    config = transformers.AutoConfig.from_pretrained(source, **common)
    tokenizer = transformers.AutoTokenizer.from_pretrained(source, **common)
    cls = transformers.AutoModelForCausalLM
    if config.model_type in TEXT_EXTRACTED_MODEL_TYPES:
        cls = getattr(transformers, "Qwen3_5ForCausalLM", None)
        if cls is None:
            raise RuntimeError("Installed transformers lacks the native Qwen3.5 model")
        config = config.get_text_config()
    model, loading = cls.from_pretrained(
        source,
        config=config,
        dtype=getattr(torch, dtype),
        device_map={"": str(target)},
        low_cpu_mem_usage=True,
        output_loading_info=True,
        **common,
    )
    if any(loading.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"Checkpoint did not load completely: {loading}")
    model.eval()
    metadata = {
        "source": source,
        "revision": revision,
        "dtype": dtype,
        "device": str(target),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }
    return model, tokenizer, metadata
