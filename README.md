# CS1-Bench (Classification System 1 Benchmark)

**Classification System 1 Benchmark**: how well do different LLMs handle the fast, typed decisions that "System One" models like TypeSafe's Jev are built for?

A System One decision takes a **state** (the evidence), a **question** (the criterion) and a fixed list of **options**, and returns a probability for each option. The model writes no text. CS1-Bench reads these probabilities directly from a model's next-token logits and measures accuracy, latency and stability across models, backends and prompt conditions.

> Thesis project. Not affiliated with or endorsed by TypeSafe. Jev and TypeSafe are trademarks of their respective owners.

## Status

| Component | State |
|---|---|
| Scoring engine (direct, serial, shared, reranker) | Imported from SemIf |
| Backends: Torch (CUDA, MPS, CPU) and llama.cpp (GGUF) | Imported from SemIf |
| Data import: CSV and other formats into `state` / `question` / `options` | Planned |
| Runner: one model at a time, row-level results plus a run manifest | Planned |
| Evaluator: accuracy, latency, stability | Planned |
| Bias conditions: system-prompt language, option position, system-role handling | Planned |

## Decision format

Each input line is one JSON object:

```json
{"id": "route-1",
 "state": "Customer asks to reset a forgotten password and says the reset email never arrived.",
 "question": "Which queue should handle this request?",
 "options": [
   {"id": "account_access", "description": "Account access and authentication support."},
   {"id": "billing", "description": "Billing and payment support."},
   {"id": "sales", "description": "Sales and product evaluation."}]}
```

`state` can be a string, object or array. A decision has 2 to 16 options, and option IDs must be unique.

## Quick start

Python 3.10+:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
pytest -q
```

Score a JSONL file with a pinned model:

```bash
CUDA_VISIBLE_DEVICES=0 cs1-score --mode direct \
  --model Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --input decisions.jsonl --output runs/qwen3.5-4b.jsonl
```

The output path must not exist yet. Other options:

- `--mode`: `direct`, `serial`, `shared` or `reranker`.
- `--device`: `cuda`, `mps` or `cpu`.
- `--backend llamacpp --gguf model.gguf`: score a GGUF checkpoint with llama.cpp (install with `pip install -e '.[test,llamacpp]'`).

## How scoring works

The model receives the decision as JSON, with options labelled `A`, `B`, `C`… CS1-Bench runs one forward pass and keeps the logits of those letter tokens, so there is no decoding or answer parsing. The tokenizer must encode each letter as exactly one token, and appending it must not change how the prompt is tokenized. Otherwise the row fails instead of producing a silently wrong score.

- `serial` and `shared` reuse the KV cache of a common `state` across questions.
- `reranker` scores each option separately with Qwen3-Reranker. The model never sees the option order, so this mode serves as a control for position bias.

## Credits

The scoring engine comes from [SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev) by TheoLeeCJ (MIT). The `semif-upstream` tag marks the imported state. CS1-Bench is released under the MIT License; see [LICENSE](LICENSE).
