# CS1-Bench (Classification System 1 Benchmark)

**Classification System 1 Benchmark**: how well do different LLMs handle the fast, typed decisions that "System One" models like TypeSafe's Jev are built for?

A System One decision takes a **state** (the evidence), a **question** (the criterion) and a fixed list of **options**, and returns a probability for each option. The model writes no text. CS1-Bench reads these probabilities directly from a model's next-token logits and measures accuracy, latency and stability across models, backends and prompt conditions.

> Thesis project. Not affiliated with or endorsed by TypeSafe. Jev and TypeSafe are trademarks of their respective owners.

## Status

| Component | State |
|---|---|
| Scoring engine (direct, serial, shared, reranker) | Imported from SemIf |
| Backends: Torch (CUDA, MPS, CPU) and llama.cpp (GGUF) | Imported from SemIf |
| Data import: CSV, JSON and JSONL into canonical decisions (`cs1 import`) | Done |
| Runner: one model at a time, row-level results plus a run manifest (`cs1 run`) | Done |
| Evaluator: accuracy, latency, stability | Planned |
| Prompt conditions: language, format, system-role handling | Done |
| Variants: prompt conditions × option rotations (`cs1 variants`) | Done |

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

`cs1 import` converts CSV, JSON (an array) or JSONL into the format above and writes a new JSONL file:

```bash
cs1 import tickets.csv --output data/tickets.jsonl --lang it
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

## Generating variants

`cs1 variants` takes imported decisions and makes one row per experimental condition:

```bash
cs1 variants data/tickets.jsonl --output data/tickets-variants.jsonl \
  --langs en,it --formats json,text --rotations cyclic
```

With `--rotations cyclic`, the options get shifted around so that every option shows up in every position exactly once:

```text
r0: A=account  B=billing  C=sales
r1: A=billing  B=sales    C=account
r2: A=sales    B=account  C=billing
```

The `answer` stays the same option ID in every rotation. If a model keeps picking `A` no matter what's behind it, that's position bias.

Each variant gets its own ID (like `d1#it.text.r2`) and remembers where it came from in `base_id`, plus its `rotation`. None of this reaches the prompt.

Watch the size: the variants multiply. 2 languages × 2 formats × 3 rotations is 12 rows per decision, so 100 decisions become 1,200 forward passes per model.

For a cheaper position-bias check, `--rotations reverse` makes just two orders per decision: the original (`r0`) and the reversed one (`rotation: "reverse"`, IDs ending in `.rev`). That's 2 passes instead of one per option.

## Quick start

Python 3.10+. PyTorch is an optional extra, because the right build depends on the machine:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[torch,test]'   # Mac or plain CPU: installs torch 2.10.0
pytest -q
cs1 --help
```

If the environment already has a working PyTorch (for example a CUDA 13 build on a DGX Spark), leave it alone and skip the extra. Do a dry run first to make sure nothing replaces it:

```bash
uv pip install --python /path/to/.venv/bin/python -e '.[test]' --dry-run   # torch must not appear here
uv pip install --python /path/to/.venv/bin/python -e '.[test]'
```

Every run manifest records the exact torch version, so nothing gets lost by not pinning it.

## Running a model

The whole pipeline is import → variants → check → run. With models already downloaded into one folder (one subfolder per model), it goes like this:

```bash
export CS1_MODELS_DIR=~/models   # then --model takes just the folder name
export HF_HUB_OFFLINE=1          # never touch the network at runtime

cs1 import tickets.csv --output data/tickets.jsonl --lang it
cs1 variants data/tickets.jsonl --output data/tickets-variants.jsonl --langs en,it --rotations cyclic
cs1 check Qwen3.5-4B --input data/tickets-variants.jsonl  # static checks + a 3-row live run
cs1 run data/tickets-variants.jsonl --mode direct --model Qwen3.5-4B --output runs/qwen3.5-4b/
```

**Revisions.** For a local folder, `--revision` is optional. CS1-Bench reads the commit from the `.cache/huggingface/download/*.metadata` files that `hf download --local-dir` leaves behind, and they all have to agree. If those files aren't there, it looks for a `REVISION` file with the hash in the model folder. If it can't find either, it stops and asks for `--revision`. The manifest records where the revision came from (`revision_source`). Hugging Face IDs like `Qwen/Qwen3.5-4B` still need `--revision`.

**`cs1 check`** works in levels. You get as much as you ask for:

```bash
cs1 check                                  # environment, plus a static look at every model in $CS1_MODELS_DIR
cs1 check Qwen3.5-4B                       # + loads that model and runs the built-in 4-row smoke test
cs1 check Qwen3.5-4B --input mine.jsonl    # + checks your file and scores its first 3 rows
cs1 check Qwen3.5-4B --no-load             # static checks only, no weights
```

The **static checks** take a few seconds per model and never load weights:

- the revision it found;
- whether the loader can handle the model;
- weight size, and whether the chat template has a system role;
- whether `A`, `B`, `C`… are single tokens;
- whether every prompt (your file's, or the smoke test's) encodes cleanly.

The **live test** runs the real `cs1 run` on those few rows, in its own process and a temporary folder, so nothing ends up in `runs/`. It shows load time, the class that was loaded, and each row's pick with its probability and the expected answer. Wrong answers are just information. Only things that break (a failed load, a row error, a manifest that isn't `completed`) count as ✗. The smoke test is four trivial decisions, one for each language and format (`src/cs1_bench/smoke.jsonl`).

Checkpoints that aren't generative language models, or whose `model_type` transformers doesn't know at all (with no `auto_map` to load it), are marked `–` (not evaluable): they need another engine or their own library. Models that ship their own code (`auto_map`) are ✗ unless you pass `--allow-remote-code`: then the static check lists their `.py` files without running them, and only the live test executes them. The exit code is 1 if anything is ✗.

`cs1 run` loads the model once, scores every row and creates a new folder with two files:

- `predictions.jsonl`: one line per variant with the option probabilities, timings, prompt conditions, and the gold `answer`. It's everything the evaluator needs.
- `manifest.json`: what was run and how. That includes the git commit of the code (and whether it had uncommitted changes), the exact command, model and revision, a hash of the input file, how many rows had each condition, timings, and the hardware.

A few things it does for you:

- **Progress on screen.** In a terminal (tmux too) there's a live bar at the bottom: `▸ 1200/2400 ██████░░░░ 50% · 38.2 rows/s · ETA 00:31 · errors 0`. In a log file (`cs1 run … 2>&1 | tee run.log`) you get a plain line every 10 seconds instead (change it with `--progress-every`), so the log stays readable. Every line starts with a mark (`✓` done, `!` warning, `✗` failed, `·` info), colours only show up in a real terminal, and `NO_COLOR=1` turns them off.
- **Warm-up.** The first pass on a GPU is always slow (kernels loading, memory allocation), so the first row is run once and thrown away. Change this with `--warmup N`.
- **Our system prompt, always.** It goes in as a system message. If a model's chat template has no system role, it's put at the top of the user message instead. Each output row says which one happened (`system_role`: `native` or `merge`), and the manifest says whether the model supports a system role.
- **Remote code, only if you ask.** Some models ship their own Python code. `--allow-remote-code` lets transformers run it (that's its `trust_remote_code`). It's off by default, prints the `.py` files it's about to run, and ends up in the manifest. Read those files first, because they run with your permissions.
- **Per-row errors.** If a row can't be scored (for example, the tokenizer splits `A` into two tokens), it gets an `error` field and the run keeps going.
- **A manifest even on a crash**, with `"status": "failed"` and the error message. It also records the GPU's name, compute capability, and memory.

Other options:

- `--mode`: `direct`, `serial`, `shared` or `reranker`.
- `--device`: `cuda`, `mps` or `cpu`.
- `--backend llamacpp --gguf model.gguf`: score a GGUF checkpoint with llama.cpp (install with `pip install -e '.[test,llamacpp]'`).

`cs1 score` is the old single-file scorer from SemIf. It's still there for quick checks, but it doesn't write a manifest.

## How scoring works

The model receives the decision with options labelled `A`, `B`, `C`… (see [Prompt conditions](#prompt-conditions) for the layouts). CS1-Bench runs one forward pass and keeps the logits of those letter tokens, so there is no decoding or answer parsing. The tokenizer must encode each letter as exactly one token, and appending it must not change how the prompt is tokenized. Otherwise the row fails instead of producing a silently wrong score.

- `serial` and `shared` reuse the KV cache of a common `state` across questions.
- `reranker` scores each option separately with Qwen3-Reranker. The model never sees the option order, so this mode serves as a control for position bias.

Each output row also has `allowed_token_mass`: how much of the model's next-token probability lands on the answer letters at all. If it's low, the model wanted to say something else (like "The" or "Based"), and its letter scores are less trustworthy.

## Prompt conditions

Two optional fields on a row change how the prompt looks. They're experiment settings rather than data, so you won't usually write them by hand. Every output row records them, together with `system_role`, which isn't a setting: `cs1 run` picks it for each model (see above).

| Field | Values | What it does |
|---|---|---|
| `prompt_lang` | `en` (default), `it` | Language of the system prompt and of all the labels around the data |
| `prompt_format` | `json` (default), `text` | `json` packs everything in one JSON object, like SemIf. `text` is a plain multiple-choice question |

This is the `text` format in English (the Italian one uses `<elementi>`, `Criterio`, `Opzioni`):

```text
<information>
Customer asks to reset a forgotten password.
</information>

Criterion: Which queue should handle this request?

Options:
A. Account access and authentication support.
B. Billing and payment support.

Answer with the letter of the best option.
```

Both formats end with that last line, so it doesn't get mixed up with the format when comparing them.

## Which models work

The rule is simple: a model works if transformers knows its `model_type` either as a causal language model or as an image-text-to-text (multimodal) model. Multimodal models are loaded whole and only ever get text. On top of that, the tokenizer needs a chat template and must encode `A`, `B`, `C`… as single tokens. `cs1 check` tests all of this in a few seconds without loading the weights.

So a new architecture usually needs no code at all. It works as soon as the installed transformers supports it. Qwen3.5 is the one special case: its text model is extracted on its own, as SemIf already did. Every manifest records the class that was actually loaded (`model_class`).

Tried so far on the DGX Spark: Qwen3.5 (0.8B to 9B), Qwen3-8B, MiniCPM5-2B, and Gemma 4 E2B. For any other model, run `cs1 check` first.

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

**Natural prompts in two languages**

SemIf's system prompt and JSON keys are only in English. My hunch is that small models might get confused when the data is in Italian but the prompt is in English. To check this properly, I track the prompt language and the data language (`lang`) separately and plan to run all four combinations (EN or IT prompt × EN or IT data). That way I can hopefully tell apart two things:

- Italian data being harder for these models in general (less Italian in their training data, more tokens per word, different grammar).
- The prompt and data languages not matching.

I first tried a literal Italian translation of SemIf's prompt, but it sounded stiff and unnatural to a native speaker. So I wrote a natural Italian prompt first and then an English one that mirrors it sentence by sentence.

**A plain-text format next to JSON**

[Rizzo Flow](https://github.com/Rizzo-AI-Academy/rizzo-flow), another SemIf-inspired project, tried several prompt variants on one 4B model (bout the same size as the models I'm testing). Listing the options as plain `A. …` lines instead of JSON made answers much more stable when the options were reordered. Their test set was small, so it's a hint rather than proof. Since position bias is one of the things I want to measure, I kept both layouts as options and can compare them.

The closing line ("Answer with the letter of the best option.") also comes from there. It repeats the instruction right where the letter probabilities are read.

## Credits

The scoring engine comes from [SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev) by TheoLeeCJ (MIT). The `semif-upstream` tag marks the imported state. CS1-Bench is released under the MIT License; see [LICENSE](LICENSE).
