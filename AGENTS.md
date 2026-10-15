# Repository instructions

CS1-Bench is a bachelor's thesis project. The maintainer is learning by building it. Explain every change you make and wait for the maintainer approval. Also, the maintainer wants to edit and commit changes himself primarily, so recheck the code and give a feedback when asked.

## Environment

- Run commands from the repository root in an isolated environment installed with `pip install -e '.[test]'` (add `llamacpp` to the extras for the GGUF backend).
- Benchmark runs target an NVIDIA DGX Spark (aarch64, one Blackwell GPU, 128 GB unified memory). Development and small checks happen on macOS (Torch MPS or llama.cpp Metal).
- Validate changes with `pytest -q`. Tests must not download models or require a GPU; mock the model where needed.

## Benchmark integrity

- Run outputs are create-only: never overwrite an existing result file. Load one model per scorer process and expose exactly one CUDA GPU to it.
- Pin every remote model to a 40-character commit revision and record it in the output.
- Prompt construction lives in `src/cs1_bench/core.py`. Backends reuse it and must not build their own prompts. Any change to prompt text, structure, or message roles needs a new `PROMPT_VERSION`.
- Record every experimental condition that can change a score (prompt language, system-role handling, option order, dtype, quantization, backend) in each output row.
- Do not commit model weights, Hugging Face caches, or third-party datasets without a redistribution grant.

## Attribution

- The scoring engine derives from SemIf (MIT). Keep the original copyright line in `LICENSE`. The `semif-upstream` git tag marks the imported state.
- CS1-Bench is not affiliated with TypeSafe. Refer to Jev only as the reference design for the task format.
