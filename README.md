# CS1-Bench (Classification System 1 Benchmark)

**Classification System 1 Benchmark**: how well do different LLMs handle the fast, typed decisions that "System One" models like TypeSafe's Jev are built for?

A System One decision takes a **state** (the evidence), a **question** (the criterion) and a fixed list of **options**, and returns a probability for each option. The model writes no text. CS1-Bench reads these probabilities directly from a model's next-token logits and measures accuracy, latency and stability across models, backends and prompt conditions.

> Thesis project. Not affiliated with or endorsed by TypeSafe. Jev and TypeSafe are trademarks of their respective owners.

## Status

| Component | State |
|---|---|
| Scoring engine (direct, serial, shared, reranker) | Imported from SemIf |
| Backends: Torch (CUDA, MPS, CPU) and llama.cpp (GGUF) | Imported from SemIf |
| Data import: CSV, JSON and JSONL into canonical decisions (`cs1-import`) | Done |
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

Optional fields:

- `answer`: the gold label, given as an option `id`.
- `lang`: the language of the data, for example `en` or `it`.
- `group_id`: groups related decisions, for example variants of the same item.
- `meta`: any other information. The model never sees `answer`, `lang`, `group_id` or `meta`.

## Importing data

`cs1-import` converts CSV, JSON (an array) or JSONL into the format above and writes a new JSONL file:

```bash
cs1-import tickets.csv --output data/tickets.jsonl --lang it
```

CSV files can give options in one of two ways:

- **One column per option**, named `option_<id>`. Empty cells mean the row has fewer options:

  ```csv
  state;question;option_si;option_no;answer
  Il deploy è riuscito.;È andato a buon fine?;Sì;No;si
  ```

- **A single `options` column** holding a JSON list.

Other details:

- The delimiter (`,`, `;` or tab) is detected automatically.
- `--map state=evidence` reads a field from a column with a different name.
- Rows without an `id` get one derived from their content, so the ID stays the same when rows are reordered.
- Unknown columns are kept in `meta`.
- Any invalid row stops the import, and the error names the record.

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

## What changes from SemIf?

SemIf shows that a small open model can mimic Jev's interface pretty fast without wasting inference resources on autoregressive text generation.
CS1-Bench builds on it with a broader question: are SLMs and LLMs good at fast decision-making tasks, like classification? And do things like prompt language or option order mess with their answers? So I changed a few things:

**What I kept**

- **The scoring engine and the Torch backend.** This is the heart of SemIf, so there was no reason to touch it.
- **The llama.cpp backend.** It lets me try quantized models (Q4, Q8…) and see whether they behave differently from BF16. It reuses the same reference tokenizer, so the comparison with Torch should be fair.
- **The reranker mode.** It looks at one option at a time and never sees their order, so I'm hoping it works as a sort of control group for position bias.

**What I removed**

- **The MLX backend.** It only runs on Apple Silicon and is written for Qwen3.5. My real runs happen on an NVIDIA DGX Spark, and on a Mac you can still use Torch MPS or llama.cpp with Metal.
- **The browser demos and the EXL3 bridge.** They're cool, but they're not what this benchmark is about.
- **SemIf's datasets and published results.** Those numbers belong to SemIf. I'd rather build my own evidence than lean on theirs.

**Gold label as an option ID instead of an index**

SemIf stores the right answer as a position (`"label": 2`). I switched to the option's ID (`"answer": "supported"`). To check position bias I shuffle the options around, and an index would then point at the wrong option unless I recomputed it every time. An ID stays correct whatever the order, so there's one less thing to get wrong.

**Prompt language as a variable (coming soon)**

SemIf's system prompt and JSON keys are only in English. My hunch is that small models might get confused when the data is in Italian but the prompt is in English. To check this properly, I track the prompt language and the data language (`lang`) separately and plan to run all four combinations (EN or IT prompt × EN or IT data). That way I can hopefully tell apart two things:

- Italian data being harder for these models in general (less Italian in their training data, more tokens per word, different grammar).
- The prompt and data languages not matching.

## Credits

The scoring engine comes from [SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev) by TheoLeeCJ (MIT). The `semif-upstream` tag marks the imported state. CS1-Bench is released under the MIT License; see [LICENSE](LICENSE).
