"""What a run writes: one row per scored variant in the thesis' common format, and the run ID.

The common format is the same for every model and wrapper of the thesis (STANDARD-DATI.md):
12 fixed columns, then the optional columns this wrapper can fill. Anything constant over a run
(model, revision, dtype, prompt version, system role) goes in the manifest, not in the rows.
"""

from __future__ import annotations

import datetime

from .core import prompt_condition

COLUMNS = ("run_id", "decision_id", "set", "prompt_lang", "prompt_format", "order",
           "option_ids", "answer", "prediction", "probabilities", "confidence", "seconds")
# Copied as they are from a scorer result when present (serial and shared modes fill fewer of them).
OPTIONAL = ("allowed_token_mass", "full_vocab_argmax_id", "forward_seconds", "prompt_sha256", "input_tokens")


def run_id(model_id: str, dataset_id: str, started: datetime.datetime) -> str:
    """model_dataset_YYYYMMDD-HHMM in UTC; IDs must not contain the underscore that separates the parts."""
    for name, value in (("model ID", model_id), ("dataset ID", dataset_id)):
        if not value or "_" in value:
            raise ValueError(f"{name} {value!r} must be nonempty and contain no underscore")
    return f"{model_id}_{dataset_id}_{started.astimezone(datetime.timezone.utc):%Y%m%d-%H%M}"


def order_label(rotation) -> str:
    """Option order as dirK (first K options moved to the end) or rev0 (reversed); no rotation means dir0."""
    if rotation is None:
        return "dir0"
    if rotation == "reverse":
        return "rev0"
    if isinstance(rotation, int) and rotation >= 0:
        return f"dir{rotation}"
    raise ValueError(f"Unknown rotation {rotation!r}")


def decision_set(row: dict, decision_id: str) -> str:
    """The set a decision belongs to: its own 'set' field, else the ID prefix before the first '-'.

    The v1 Cosman decisions have no 'set' field, so their IDs give 'dom' and 'task'.
    """
    return row.get("set") or decision_id.split("-", 1)[0]


def most_likely(probabilities: dict, option_ids: list) -> str:
    """Option with the highest probability; on a tie the first in the order shown wins."""
    best = option_ids[0]
    for option in option_ids:
        if probabilities[option] > probabilities[best]:
            best = option
    return best


def common_row(run: str, row: dict, result: dict) -> dict:
    """Common-format row for one input row and its scorer result (a result with 'error' failed)."""
    decision_id = row.get("base_id", row["id"])
    option_ids = [option["id"] for option in row["options"]]  # In the order the model saw them.
    condition = prompt_condition(row)
    record = dict.fromkeys(COLUMNS)
    record.update(run_id=run, decision_id=decision_id, set=decision_set(row, decision_id),
                  prompt_lang=condition["prompt_lang"], prompt_format=condition["prompt_format"],
                  order=order_label(row.get("rotation")), option_ids=option_ids, answer=row.get("answer"))
    if "error" in result:
        record["error"] = result["error"]
        return record
    if result["option_ids"] != option_ids:
        raise ValueError(f"Row {row['id']}: scorer returned options in a different order")
    probabilities = dict(zip(option_ids, result["probabilities"]))
    record.update(prediction=most_likely(probabilities, option_ids), probabilities=probabilities,
                  seconds=result.get("total_seconds"))
    record["option_logits"] = dict(zip(option_ids, result["option_logits"]))
    record.update({field: result[field] for field in OPTIONAL if field in result})
    return record
