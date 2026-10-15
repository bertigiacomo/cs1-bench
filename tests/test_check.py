import json
import sys

import pytest

from cs1_bench import check
from cs1_bench.check import FAIL, OK, SKIP

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
      "text_config": {}, "vision_config": {}}, FAIL, "multimodal gemma4"),
    ({"model_type": "mistral3", "architectures": ["Mistral3ForConditionalGeneration"],
      "text_config": {}, "vision_config": {}}, FAIL, "multimodal mistral3"),
    ({"model_type": "sparkx", "architectures": ["SparkXForCausalLM"],
      "auto_map": {"AutoModelForCausalLM": "modeling_sparkx.SparkXForCausalLM"}}, FAIL, "auto_map"),
    ({"model_type": "minicpm5", "architectures": ["MiniCPM5ForCausalLM"]}, FAIL, "unknown to transformers"),
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
    make_model(tmp_path, "gemma-4-E2B-it", {"model_type": "gemma4", "architectures": ["Gemma4ForConditionalGeneration"],
                                            "text_config": {}, "vision_config": {}})
    with pytest.raises(SystemExit) as error:
        run_main(monkeypatch, str(tmp_path / "gemma-4-E2B-it"))
    assert error.value.code == 1
    assert "✗ loader unsupported: multimodal gemma4" in capsys.readouterr().out
