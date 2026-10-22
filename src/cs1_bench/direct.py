"""Direct categorical decision readout from native next-token logits."""

from __future__ import annotations

import inspect
import time

from .core import LETTERS, digest, direct_messages, prompt_condition, softmax, synchronize

# v2: natural en/it wording, json or text layout, native or merged system turn, and a closing line.
# v3: the text layout labels the question block "Criterion"/"Criterio" (was "Question"/"Domanda"),
#     matching the system prompt and the JSON "criterion"/"criterio" key.
PROMPT_VERSION = "cs1bench-v1"  # was direct-options-v3: same text, renamed to the registry ID


def _slot_ids(tokenizer, count: int) -> list[int]:
    result = []
    for letter in LETTERS[:count]:
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) != 1 or tokenizer.decode(encoded) != letter:
            raise ValueError(f"Answer slot {letter!r} is not one exact round-trip token")
        result.append(encoded[0])
    if len(result) != len(set(result)):
        raise ValueError("Answer-slot tokens collide")
    return result


def _forward(model, inputs):
    parameters = inspect.signature(model.forward).parameters
    kwargs = dict(inputs, use_cache=False, return_dict=True)
    if "logits_to_keep" in parameters:
        kwargs["logits_to_keep"] = 1
    return model(**kwargs).logits[:, -1, :]


def encode_prompt(tokenizer, row: dict, max_tokens: int) -> tuple[list[int], list[int], str]:
    """Encode one decision and verify its single-token answer slots."""
    prompt = tokenizer.apply_chat_template(
        direct_messages(row), tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not ids or len(ids) > max_tokens:
        raise ValueError(f"Row {row['id']}: {len(ids)} input tokens exceed limit {max_tokens}; no truncation allowed")
    slots = _slot_ids(tokenizer, len(row["options"]))
    for letter, token in zip(LETTERS, slots):
        if tokenizer.encode(prompt + letter, add_special_tokens=False) != ids + [token]:
            raise ValueError(f"Answer boundary changes tokenization for slot {letter}")
    return ids, slots, digest(prompt)


def score(model, tokenizer, row: dict, metadata: dict, max_tokens: int = 4096) -> dict:
    import torch

    started = time.perf_counter()
    ids, slots, prompt_hash = encode_prompt(tokenizer, row, max_tokens)
    device = next(model.parameters()).device
    inputs = {
        "input_ids": torch.tensor([ids], dtype=torch.long, device=device),
        "attention_mask": torch.ones((1, len(ids)), dtype=torch.long, device=device),
    }
    synchronize(device)
    forward_start = time.perf_counter()
    with torch.inference_mode():
        vocabulary = _forward(model, inputs)[0].float()
    synchronize(device)
    selected = vocabulary[slots].cpu().tolist()
    forward_seconds = time.perf_counter() - forward_start
    return {
        "id": row["id"],
        "option_ids": [option["id"] for option in row["options"]],
        "probabilities": softmax(selected),
        "option_logits": selected,
        "input_tokens": len(ids),
        # Share of the full next-token distribution that falls on the answer letters.
        "allowed_token_mass": float((vocabulary[slots].logsumexp(-1) - vocabulary.logsumexp(-1)).exp()),
        "full_vocab_argmax_id": int(vocabulary.argmax()),
        "forward_seconds": forward_seconds,
        "total_seconds": time.perf_counter() - started,
        "prompt_sha256": prompt_hash,
        "prompt_version": PROMPT_VERSION,
        **prompt_condition(row),
        "model": metadata,
        "readout": "native full-vocabulary last-position logits restricted to declared answer slots",
        "probability_status": "conditional option score; uncalibrated as decision confidence",
    }
