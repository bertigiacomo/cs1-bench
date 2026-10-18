"""Model, backend and mode options shared by cs1 run and cs1 check, and the model loader behind them."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

from .core import load_causal_model, resolve_model_source, resolve_revision
from .direct import score as direct_score
from .reranker import score as reranker_score
from .serial import SerialPrefixScorer
from .shared import score_shared

MODES = ("direct", "serial", "shared", "reranker")


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the model, backend, and scoring-mode options shared by cs1 run and cs1 check."""
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--backend", choices=("torch", "llamacpp"), default="torch")
    parser.add_argument("--gguf", type=Path, help="Local GGUF checkpoint for --backend llamacpp")
    parser.add_argument("--llama-threads", type=int,
                        help="CPU threads for --backend llamacpp (default: all visible cores)")
    parser.add_argument("--model", required=True,
                        help="Hugging Face ID, local folder, or folder name under $CS1_MODELS_DIR")
    parser.add_argument("--revision",
                        help="Commit hash; optional for local folders (read from hf download metadata or REVISION)")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto",
                        help="Torch device (auto prefers CUDA, then Apple MPS; CPU must be explicit)")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16",
                        help="Model precision; changing it can change option scores")
    parser.add_argument("--allow-remote-code", action="store_true",
                        help="Run the Python code shipped in the model folder (transformers trust_remote_code); "
                             "read those .py files first")


def check_model_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Resolve the model and its revision, and reject invalid combinations before anything is loaded."""
    args.model = resolve_model_source(args.model)
    try:
        args.revision, args.revision_source = resolve_revision(args.model, args.revision)
    except ValueError as error:
        parser.error(str(error))
    if args.max_tokens < 1:
        parser.error("max-tokens must be positive")
    if args.gguf is not None and args.backend != "llamacpp":
        parser.error("--gguf requires --backend llamacpp")
    if args.llama_threads is not None:
        if args.backend != "llamacpp":
            parser.error("--llama-threads requires --backend llamacpp")
        if args.llama_threads < 1:
            parser.error("--llama-threads must be positive")
    if args.backend == "llamacpp":
        if args.allow_remote_code:
            parser.error("--allow-remote-code applies to --backend torch only")
        if args.mode == "reranker":
            parser.error("llama.cpp supports direct, serial, and shared modes; reranker requires torch")
        if args.gguf is None or not args.gguf.is_file():
            parser.error("--backend llamacpp requires --gguf pointing at an existing GGUF file")
    elif args.mode == "reranker":
        if args.device in {"mps", "cpu"}:
            parser.error(f"Reranker mode requires CUDA; --device {args.device} is unsupported")
        # Keep auto CUDA-only and let the loader enforce one visible GPU.
        args.device = "cuda"


def load_backend(args: argparse.Namespace):
    """Load one model and return it with the scorer functions of its backend."""
    if args.backend == "llamacpp":
        from . import llamacpp_backend

        model, tokenizer, metadata = llamacpp_backend.load_model(
            args.model, args.revision, args.gguf,
            threads=args.llama_threads, context_tokens=args.max_tokens)
        scorers = SimpleNamespace(direct=llamacpp_backend.score, serial=llamacpp_backend.SerialPrefixScorer,
                                  shared=llamacpp_backend.score_shared, reranker=None)
    else:
        model, tokenizer, metadata = load_causal_model(args.model, args.revision, args.device, args.dtype,
                                                       trust_remote_code=args.allow_remote_code)
        scorers = SimpleNamespace(direct=direct_score, serial=SerialPrefixScorer,
                                  shared=score_shared, reranker=reranker_score)
    return model, tokenizer, metadata, scorers

