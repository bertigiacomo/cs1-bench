"""Create-only JSONL command line scorer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import load_causal_model, validate_row
from .direct import score as direct_score
from .reranker import score as reranker_score
from .serial import SerialPrefixScorer
from .shared import score_shared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("direct", "serial", "shared", "reranker"), required=True)
    parser.add_argument("--backend", choices=("torch", "llamacpp"), default="torch")
    parser.add_argument("--gguf", type=Path, help="Local GGUF checkpoint for --backend llamacpp")
    parser.add_argument("--llama-threads", type=int,
                        help="CPU threads for --backend llamacpp (default: all visible cores)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto",
                        help="Torch device (auto prefers CUDA, then Apple MPS; CPU must be explicit)")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16",
                        help="Model precision; changing it can change option scores")
    args = parser.parse_args()
    if args.output.exists() or args.max_tokens < 1:
        parser.error("Output must be new and max-tokens must be positive")
    if args.gguf is not None and args.backend != "llamacpp":
        parser.error("--gguf requires --backend llamacpp")
    if args.llama_threads is not None:
        if args.backend != "llamacpp":
            parser.error("--llama-threads requires --backend llamacpp")
        if args.llama_threads < 1:
            parser.error("--llama-threads must be positive")
    if args.backend == "llamacpp":
        if args.mode == "reranker":
            parser.error("llama.cpp supports direct, serial, and shared modes; reranker requires torch")
        if args.gguf is None or not args.gguf.is_file():
            parser.error("--backend llamacpp requires --gguf pointing at an existing GGUF file")
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if not rows:
        parser.error("Input is empty")
    for row in rows:
        validate_row(row)
    direct, serial, shared = direct_score, SerialPrefixScorer, score_shared
    if args.backend == "llamacpp":
        from . import llamacpp_backend

        model, tokenizer, metadata = llamacpp_backend.load_model(
            args.model, args.revision, args.gguf,
            threads=args.llama_threads, context_tokens=args.max_tokens)
        direct, serial, shared = (llamacpp_backend.score, llamacpp_backend.SerialPrefixScorer,
                                  llamacpp_backend.score_shared)
    else:
        if args.mode == "reranker":
            if args.device in {"mps", "cpu"}:
                parser.error(f"Reranker mode requires CUDA; --device {args.device} is unsupported")
            # Keep auto CUDA-only and let the loader enforce one visible GPU.
            args.device = "cuda"
        model, tokenizer, metadata = load_causal_model(args.model, args.revision, args.device, args.dtype)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as destination:
        if args.mode == "shared":
            results, timing = shared(model, tokenizer, rows, metadata, args.max_tokens)
            for result in results:
                destination.write(json.dumps({**result, "shared_timing": timing}, allow_nan=False) + "\n")
        elif args.mode == "serial":
            scorer = serial(model, tokenizer, metadata, args.max_tokens)
            for row in rows:
                destination.write(json.dumps(scorer.score(row), allow_nan=False) + "\n")
                destination.flush()
        else:
            scorer = direct if args.mode == "direct" else reranker_score
            for row in rows:
                destination.write(json.dumps(scorer(model, tokenizer, row, metadata, args.max_tokens), allow_nan=False) + "\n")
                destination.flush()


if __name__ == "__main__":
    main()
