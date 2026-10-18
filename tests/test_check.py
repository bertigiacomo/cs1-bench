import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cs1_bench import check, ui
from cs1_bench.check import FAIL, OK, SKIP
from cs1_bench.data import validate_decision

DECISION = {
    "id": "d1",
    "state": "Customer cannot log in.",
    "question": "Which queue?",
    "options": [{"id": "account", "description": "Account access."}, {"id": "billing", "description": "Billing."}],
}


@pytest.mark.parametrize("config,status,words", [
    ({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}, OK, "Qwen3ForCausalLM"),
    ({"model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"],
      "text_config": {}, "vision_config": {}}, OK, "text model of qwen3_5"),
    ({"model_type": "gemma4", "architectures": ["Gemma4ForConditionalGeneration"],
      "text_config": {}, "vision_config": {}}, OK, "via AutoModelForCausalLM (multimodal, fed text only)"),
    ({"model_type": "mistral3", "architectures": ["Mistral3ForConditionalGeneration"],
      "text_config": {}, "vision_config": {}}, OK, "Mistral3ForConditionalGeneration via AutoModelForImageTextToText"),
    ({"model_type": "sparkx", "architectures": ["SparkXForCausalLM"],
      "auto_map": {"AutoModelForCausalLM": "modeling_sparkx.SparkXForCausalLM"}}, FAIL, "auto_map"),
    ({"model_type": "clm", "architectures": ["CLMForCausalLM"]}, SKIP, "unknown to transformers"),
    ({"model_type": "bert", "architectures": ["BertForMaskedLM"]}, SKIP, "does not generate text"),
    (None, SKIP, "no config.json"),
])
def test_classify_checkpoint(config, status, words):
    found, reason = check.classify_checkpoint(config)
    assert found == status and words in reason


class Tokenizer:
    chat_template = "fake"

    def apply_chat_template(self, turns, **kwargs):
        return "\n".join(turn["content"] for turn in turns) + "\nASSISTANT\n"

    def encode(self, text, add_special_tokens=False):
        return list(text.encode())

    def decode(self, ids):
        return bytes(ids).decode()


def make_model(root, name, config=None, commit="a" * 40):
    folder = root / name
    folder.mkdir()
    if config is not None:
        (folder / "config.json").write_text(json.dumps(config))
        (folder / "model.safetensors").write_bytes(b"0" * 1024)
        (folder / "REVISION").write_text(commit)
    return folder


def run_main(monkeypatch, *argv):
    monkeypatch.setattr(check, "load_tokenizer", lambda folder: Tokenizer())
    monkeypatch.setattr(sys, "argv", ["cs1 check", *argv])
    check.main()


def test_check_encodes_real_prompts_and_summarizes(tmp_path, monkeypatch, capsys):
    models = tmp_path / "models"
    models.mkdir()
    make_model(models, "Qwen3-8B", {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]})
    make_model(models, "laya")  # No config.json: not a transformers checkpoint.
    variants = tmp_path / "variants.jsonl"
    variants.write_text("".join(json.dumps(dict(DECISION, id=f"d{index}", prompt_format=fmt)) + "\n"
                                for index, fmt in enumerate(["json", "text"])))
    run_main(monkeypatch, "--models-dir", str(models), "--input", str(variants))
    out = capsys.readouterr().out
    assert "revision aaaaaaaaaaaa… (revision-file)" in out
    assert "letters A-B are single tokens" in out
    assert "all 2 rows encode in 2 prompt conditions" in out
    assert "– not evaluable with this engine: no config.json" in out
    assert "1 ready" in out and "1 not evaluable" in out


def test_check_exits_nonzero_when_a_model_is_unsupported(tmp_path, monkeypatch, capsys):
    make_model(tmp_path, "Spark-X2.5-4B", {"model_type": "sparkx", "architectures": ["SparkXForCausalLM"],
                                           "auto_map": {"AutoModelForCausalLM": "modeling_spark.SparkXForCausalLM"}})
    with pytest.raises(SystemExit) as error:
        run_main(monkeypatch, str(tmp_path / "Spark-X2.5-4B"))
    assert error.value.code == 1
    assert "✗ requires remote code (auto_map" in capsys.readouterr().out


def test_allow_remote_code_turns_auto_map_into_a_listed_warning(tmp_path, monkeypatch, capsys):
    folder = make_model(tmp_path, "Spark-X2.5-4B", {"model_type": "sparkx", "architectures": ["SparkXForCausalLM"],
                                                    "auto_map": {"AutoModelForCausalLM": "modeling_spark.SparkX"}})
    (folder / "modeling_spark.py").write_text("x" * 2048)
    run_main(monkeypatch, "--allow-remote-code", "--no-load", str(folder))  # Warnings are not failures.
    out = capsys.readouterr().out
    assert "! remote code allowed (auto_map" in out
    assert "! code that would run: modeling_spark.py (2 KiB)" in out


def test_smoke_file_is_tiny_valid_and_covers_both_languages_and_formats():
    rows = check.read_rows(check.SMOKE)
    for row in rows:
        validate_decision(row)
    assert len(rows) <= 5
    assert {(row["prompt_lang"], row["prompt_format"]) for row in rows} == {
        ("en", "json"), ("en", "text"), ("it", "json"), ("it", "text")}


REAL_POPEN = subprocess.Popen


def fake_popen(status="completed", error=None, calls=None):
    """Stand-in for subprocess.Popen running cs1 run: writes its outputs and emits both kinds of stderr lines.

    Other commands (git, which subprocess.run also starts through Popen) pass through unchanged.
    """
    def popen(command, **kwargs):
        if "cs1_bench" not in command:
            return REAL_POPEN(command, **kwargs)
        if calls is not None:
            calls.append((command, kwargs))
        source = Path(command[command.index("run") + 1])
        output = Path(command[command.index("--output") + 1])
        output.mkdir()
        rows = check.read_rows(source)
        manifest = {"status": status, "error": error, "backend": "torch", "load_seconds": 1.5,
                    "scoring_seconds": 0.5, "wall_seconds": 2.2, "rows_scored": len(rows), "rows_failed": 0,
                    "system_role_supported": True, "model": {"model_class": "Qwen3ForCausalLM", "device": "cuda:0"}}
        (output / "manifest.json").write_text(json.dumps(manifest))
        (output / "predictions.jsonl").write_text("".join(json.dumps(
            {"id": row["id"], "option_ids": [o["id"] for o in row["options"]], "probabilities": [0.9, 0.1],
             "answer": row.get("answer"), "system_role": "native"}) + "\n" for row in rows))
        stderr = ["  · loading model (torch, direct)\n",
                  "Loading weights:  50%|█████     | 1/2 [00:01<00:01,  1.00it/s]\n",
                  "[transformers] tokenizer regex warning\n",
                  "  ✓ done in 0:00:02, /tmp/run\n"]
        return SimpleNamespace(stderr=iter(stderr), wait=lambda: 0, returncode=0)
    return popen


QWEN3 = {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}


def test_named_model_without_input_runs_the_built_in_smoke_test(tmp_path, monkeypatch, capsys):
    calls = []
    folder = make_model(tmp_path, "Qwen3-8B", QWEN3)
    monkeypatch.setattr(ui.subprocess, "Popen", fake_popen(calls=calls))
    run_main(monkeypatch, str(folder))
    out = capsys.readouterr().out
    assert "✓ loaded Qwen3ForCausalLM in 1.5s on cuda:0" in out
    assert "✓ scored 4 rows in 0.5s (8.0 rows/s), 0 errors, system role native" in out
    assert "✓ smoke test done in 0:00:02: 3/4 as expected" in out  # Wrong answers inform, they do not fail.
    assert "[transformers] tokenizer regex warning" in out  # Kept: not a progress line.
    assert "loading model" not in out and "Loading weights" not in out and "/tmp/run" not in out
    assert "smoke-en-json" not in out  # No per-decision lines: they are in predictions.jsonl.
    command, kwargs = calls[0]
    assert command[command.index("--model") + 1] == str(folder)
    assert kwargs["env"]["CS1_NESTED"] == "1"


def test_input_rows_are_all_encoded_but_only_the_first_three_are_scored(tmp_path, monkeypatch, capsys):
    calls = []
    folder = make_model(tmp_path, "Qwen3-8B", QWEN3)
    variants = tmp_path / "mine.jsonl"
    variants.write_text("".join(json.dumps(dict(DECISION, id=f"d{index}")) + "\n" for index in range(5)))
    monkeypatch.setattr(ui.subprocess, "Popen", fake_popen(calls=calls))
    run_main(monkeypatch, str(folder), "--input", str(variants), "--allow-remote-code")
    out = capsys.readouterr().out
    assert "mine.jsonl: 5 valid rows" in out and "all 5 rows encode" in out
    assert "scored 3 rows" in out and "first 3 rows of mine.jsonl done in" in out
    assert "--allow-remote-code" in calls[0][0]


def test_a_failed_live_run_fails_the_check(tmp_path, monkeypatch, capsys):
    folder = make_model(tmp_path, "Qwen3-8B", QWEN3)
    monkeypatch.setattr(ui.subprocess, "Popen",
                        fake_popen(status="failed", error="RuntimeError: Checkpoint did not load completely"))
    with pytest.raises(SystemExit) as error:
        run_main(monkeypatch, str(folder))
    assert error.value.code == 1
    assert "✗ live run failed: RuntimeError: Checkpoint did not load completely" in capsys.readouterr().out


def test_no_load_and_unnamed_models_never_start_a_live_run(tmp_path, monkeypatch):
    make_model(tmp_path, "Qwen3-8B", QWEN3)
    calls = []
    monkeypatch.setattr(ui.subprocess, "Popen", fake_popen(calls=calls))
    run_main(monkeypatch, "--no-load", str(tmp_path / "Qwen3-8B"))
    run_main(monkeypatch, "--models-dir", str(tmp_path))
    assert calls == []


def test_without_models_only_the_environment_is_checked(monkeypatch, capsys):
    monkeypatch.delenv("CS1_MODELS_DIR", raising=False)
    run_main(monkeypatch)
    out = capsys.readouterr().out
    assert "Environment" in out and "Summary" not in out
