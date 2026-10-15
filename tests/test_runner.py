import argparse
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from cs1_bench import runner
from cs1_bench.conditions import expand

DECISION = {
    "id": "d1",
    "state": "Customer cannot log in.",
    "question": "Which queue?",
    "options": [{"id": "account", "description": "Account access."}, {"id": "billing", "description": "Billing."}],
    "answer": "account",
    "lang": "en",
}


class Tokenizer:
    def __init__(self, system_role=True):
        self.system_role = system_role

    def apply_chat_template(self, turns, **kwargs):
        if not self.system_role and turns[0]["role"] == "system":
            raise ValueError("System role not supported")
        return "\n".join(turn["content"] for turn in turns)


def fake_result(row):
    return {"id": row["id"], "probabilities": [0.75, 0.25], "prompt_lang": row.get("prompt_lang", "en"),
            "system_role": row.get("system_role", "native")}


@pytest.fixture
def backend(monkeypatch):
    calls = SimpleNamespace(direct=[], shared=[], tokenizer=Tokenizer())

    def direct(model, tokenizer, row, metadata, max_tokens):
        calls.direct.append(row)
        if row.get("fail"):
            raise ValueError("Answer slot 'A' is not one exact round-trip token")
        return fake_result(row)

    def shared(model, tokenizer, rows, metadata, max_tokens):
        calls.shared.append([row["id"] for row in rows])
        return [fake_result(row) for row in rows], {"total_seconds": 0.5}

    scorers = SimpleNamespace(direct=direct, shared=shared, serial=None, reranker=direct)
    monkeypatch.setattr(runner, "load_backend",
                        lambda args: ("model", calls.tokenizer, {"source": "fake", "revision": "a" * 40}, scorers))
    return calls


def make_args(mode="direct", warmup=0):
    return argparse.Namespace(mode=mode, backend="torch", model="fake", revision="a" * 40, max_tokens=128,
                              warmup=warmup, progress_every=0, device="cpu", dtype="float32", gguf=None,
                              llama_threads=None)


def write_rows(tmp_path, rows):
    path = tmp_path / "variants.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def read_output(output):
    predictions = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    return predictions, json.loads((output / "manifest.json").read_text())


def test_run_writes_self_contained_predictions_and_manifest(tmp_path, backend):
    rows = expand(DECISION, langs=("en", "it"), rotations="cyclic")
    source = write_rows(tmp_path, rows)
    runner.run(rows, tmp_path / "out", make_args(), source)
    predictions, manifest = read_output(tmp_path / "out")
    assert [prediction["id"] for prediction in predictions] == [row["id"] for row in rows]
    first = predictions[0]
    assert (first["answer"], first["base_id"], first["rotation"], first["lang"]) == ("account", "d1", 0, "en")
    assert manifest["status"] == "completed"
    assert (manifest["rows_scored"], manifest["rows_failed"]) == (4, 0)
    assert manifest["input"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert manifest["model"]["revision"] == "a" * 40
    assert manifest["conditions"]["prompt_lang"] == {"en": 2, "it": 2}
    assert manifest["system_role_supported"] is True
    assert {"git_commit", "git_dirty"} <= manifest["code"].keys()


def test_models_without_system_role_fall_back_to_merge_and_say_so(tmp_path, backend):
    backend.tokenizer = Tokenizer(system_role=False)
    rows = expand(DECISION, roles=("native", "merge"))
    runner.run(rows, tmp_path / "out", make_args(), write_rows(tmp_path, rows))
    predictions, manifest = read_output(tmp_path / "out")
    assert all(row["system_role"] == "merge" for row in backend.direct)
    native, merged = predictions
    assert (native["system_role"], native["system_role_requested"]) == ("merge", "native")
    assert "system_role_requested" not in merged
    assert (manifest["system_role_supported"], manifest["system_role_fallbacks"]) == (False, 1)


def test_a_failing_row_is_recorded_and_the_run_continues(tmp_path, backend):
    rows = [dict(DECISION, id="bad", fail=True), dict(DECISION, id="good")]
    runner.run(rows, tmp_path / "out", make_args(), write_rows(tmp_path, rows))
    predictions, manifest = read_output(tmp_path / "out")
    assert "not one exact round-trip token" in predictions[0]["error"]
    assert predictions[0]["answer"] == "account"
    assert "error" not in predictions[1]
    assert (manifest["rows_scored"], manifest["rows_failed"]) == (1, 1)


def test_warmup_runs_untimed_and_is_not_written(tmp_path, backend):
    rows = [dict(DECISION, id="one"), dict(DECISION, id="two")]
    runner.run(rows, tmp_path / "out", make_args(warmup=2), write_rows(tmp_path, rows))
    predictions, manifest = read_output(tmp_path / "out")
    assert [row["id"] for row in backend.direct] == ["one", "one", "one", "two"]
    assert len(predictions) == 2 and manifest["warmup_repeats"] == 2


def test_shared_mode_groups_by_state_and_prompt_condition(tmp_path, backend):
    questions = [dict(DECISION, id=f"q{index}", question=f"Question {index}?") for index in range(2)]
    rows = [variant for question in questions for variant in expand(question, langs=("en", "it"))]
    runner.run(rows, tmp_path / "out", make_args(mode="shared"), write_rows(tmp_path, rows))
    assert backend.shared == [["q0#en.json.native.r0", "q1#en.json.native.r0"],
                              ["q0#it.json.native.r0", "q1#it.json.native.r0"]]
    predictions, _ = read_output(tmp_path / "out")
    assert all(prediction["shared_timing"] == {"total_seconds": 0.5} for prediction in predictions)


def test_reranker_predictions_do_not_claim_prompt_conditions(tmp_path, backend):
    rows = expand(DECISION, langs=("it",))
    runner.run(rows, tmp_path / "out", make_args(mode="reranker"), write_rows(tmp_path, rows))
    [prediction], manifest = read_output(tmp_path / "out")
    assert prediction["base_id"] == "d1"
    assert "system_role_requested" not in prediction
    assert "system_role_supported" not in manifest


def test_failed_load_still_leaves_a_manifest(tmp_path, monkeypatch):
    def broken(args):
        raise RuntimeError("Checkpoint did not load completely")

    monkeypatch.setattr(runner, "load_backend", broken)
    rows = [DECISION]
    with pytest.raises(RuntimeError):
        runner.run(rows, tmp_path / "out", make_args(), write_rows(tmp_path, rows))
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "Checkpoint did not load completely" in manifest["error"]


def test_cli_refuses_an_existing_output_directory(tmp_path, monkeypatch, capsys):
    source = write_rows(tmp_path, [DECISION])
    (tmp_path / "out").mkdir()
    monkeypatch.setattr(sys, "argv", ["cs1-run", str(source), "--output", str(tmp_path / "out"),
                                      "--mode", "direct", "--model", "fake", "--revision", "a" * 40])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert "Output directory must be new" in capsys.readouterr().err


def test_progress_prints_one_line_per_interval_with_rate_and_eta():
    now = [0.0]
    lines = []
    progress = runner.Progress("Qwen3.5-4B", total=100, every=10, clock=lambda: now[0],
                               stream=SimpleNamespace(write=lines.append, flush=lambda: None))
    now[0] = 5
    progress.advance([{}] * 10)
    assert lines == []  # Interval not reached yet.
    now[0] = 10
    progress.advance([{}] * 9 + [{"error": "x"}])
    assert lines[0] == "[Qwen3.5-4B] 20/100 (20%) · 2.0 rows/s · ETA 0:00:40 · errors 1"


def test_run_reports_progress_on_stderr(tmp_path, backend, capsys):
    rows = [DECISION]
    runner.run(rows, tmp_path / "out", make_args(), write_rows(tmp_path, rows))
    err = capsys.readouterr().err
    assert "[fake] loading fake" in err and "1/1 (100%)" in err and "done in" in err
