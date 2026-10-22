import argparse
import io
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from cs1_bench import runner, ui
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
    return {"id": row["id"], "option_ids": [option["id"] for option in row["options"]],
            "probabilities": [0.75, 0.25], "option_logits": [2.0, 0.9], "total_seconds": 0.1,
            "prompt_lang": row.get("prompt_lang", "en"), "system_role": row.get("system_role", "native")}


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

    scorers = SimpleNamespace(direct=direct, shared=shared, serial=None)
    monkeypatch.setattr(runner, "load_backend",
                        lambda args: ("model", calls.tokenizer, {"device": "cpu", "model_class": "Fake"}, scorers))
    monkeypatch.setattr(runner, "_git_state", lambda: {"git_commit": "c" * 40, "git_dirty": False})
    return calls


def make_args(mode="direct", warmup=0, model="fake", allow_remote_code=False, run_id="fake_variants_20261022-2000"):
    return argparse.Namespace(mode=mode, backend="torch", model=model, revision="a" * 40, max_tokens=128,
                              warmup=warmup, progress_every=0, revision_source="argument",
                              allow_remote_code=allow_remote_code, device="cpu", dtype="float32", gguf=None,
                              llama_threads=None, model_id=None, run_id=run_id)


def write_rows(tmp_path, rows):
    path = tmp_path / "variants.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def read_output(output, run_id="fake_variants_20261022-2000"):
    predictions = [json.loads(line) for line in (output / f"{run_id}.predictions.jsonl").read_text().splitlines()]
    return predictions, json.loads((output / f"{run_id}.manifest.json").read_text())


# The 24 manifest fields of the thesis (STANDARD-DATI.md), in order, then the ones only cs1-bench writes.
STANDARD_FIELDS = ["run_id", "status", "manifest_origin", "model_id", "wrapper_id", "prompt_id", "python_version",
                   "model_revision", "dtype", "code_commit", "inputs", "prompt_langs", "prompt_formats", "orders",
                   "rows_written", "rows_failed", "started_utc", "probabilities_kind", "confidence_kind",
                   "seconds_kind", "temperature", "torch_version", "transformers_version", "gpu"]


def test_run_writes_common_rows_and_the_standard_manifest(tmp_path, backend):
    rows = expand(DECISION, langs=("en", "it"), rotations="cyclic")
    source = write_rows(tmp_path, rows)
    runner.run(rows, tmp_path / "out", make_args(), source)
    predictions, manifest = read_output(tmp_path / "out")
    assert [(p["prompt_lang"], p["order"]) for p in predictions] == [("en", "dir0"), ("en", "dir1"),
                                                                      ("it", "dir0"), ("it", "dir1")]
    first, second = predictions[0], predictions[1]
    assert list(first)[:12] == ["run_id", "decision_id", "set", "prompt_lang", "prompt_format", "order",
                                "option_ids", "answer", "prediction", "probabilities", "confidence", "seconds"]
    assert (first["run_id"], first["decision_id"], first["set"]) == ("fake_variants_20261022-2000", "d1", "d1")
    assert first["probabilities"] == {"account": 0.75, "billing": 0.25} and first["prediction"] == "account"
    assert second["option_ids"] == ["billing", "account"] and second["prediction"] == "billing"  # As seen.
    assert first["option_logits"] == {"account": 2.0, "billing": 0.9} and first["confidence"] is None
    assert list(manifest)[:24] == STANDARD_FIELDS
    assert manifest["status"] == "completed" and manifest["prompt_id"] == "cs1bench-v1"
    assert (manifest["rows_written"], manifest["rows_failed"]) == (4, 0)
    assert manifest["inputs"] == [{"dataset_id": "variants", "rows": 4,
                                   "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}]
    assert (manifest["prompt_langs"], manifest["orders"]) == (["en", "it"], ["dir0", "dir1"])
    assert (manifest["model_id"], manifest["model_revision"], manifest["code_commit"]) == ("fake", "a" * 40, "c" * 40)
    assert (manifest["code_dirty"], manifest["system_role"]) == (False, "native")


def test_run_id_comes_from_model_dataset_and_utc_time(tmp_path, backend):
    rows = [DECISION]
    source = write_rows(tmp_path, rows)
    runner.run(rows, tmp_path / "out", make_args(model="models/Qwen3.5-9B", run_id=None), source)
    [manifest] = (tmp_path / "out").glob("*.manifest.json")
    assert manifest.name.startswith("Qwen3.5-9B_variants_") and len(manifest.name) == len(
        "Qwen3.5-9B_variants_20261022-2000.manifest.json")


def test_existing_run_files_are_never_overwritten(tmp_path, backend):
    rows = [DECISION]
    source = write_rows(tmp_path, rows)
    runner.run(rows, tmp_path / "out", make_args(), source)
    with pytest.raises(FileExistsError):
        runner.run(rows, tmp_path / "out", make_args(), source)


def test_models_without_system_role_fall_back_to_merge_and_say_so(tmp_path, backend):
    backend.tokenizer = Tokenizer(system_role=False)
    rows = expand(DECISION, langs=("en", "it"))
    runner.run(rows, tmp_path / "out", make_args(), write_rows(tmp_path, rows))
    predictions, manifest = read_output(tmp_path / "out")
    assert all(row["system_role"] == "merge" for row in backend.direct)
    assert all("system_role" not in prediction for prediction in predictions)  # Constant: in the manifest.
    assert manifest["system_role"] == "merge"


def test_a_failing_row_is_recorded_and_the_run_continues(tmp_path, backend):
    rows = [dict(DECISION, id="bad", fail=True), dict(DECISION, id="good")]
    runner.run(rows, tmp_path / "out", make_args(), write_rows(tmp_path, rows))
    predictions, manifest = read_output(tmp_path / "out")
    assert "not one exact round-trip token" in predictions[0]["error"]
    assert (predictions[0]["answer"], predictions[0]["prediction"], predictions[0]["probabilities"]) == (
        "account", None, None)
    assert "error" not in predictions[1]
    assert (manifest["status"], manifest["rows_written"], manifest["rows_failed"]) == ("partial", 2, 1)


def test_warmup_runs_untimed_and_is_not_written(tmp_path, backend):
    rows = [dict(DECISION, id="one"), dict(DECISION, id="two")]
    runner.run(rows, tmp_path / "out", make_args(warmup=2), write_rows(tmp_path, rows))
    predictions, manifest = read_output(tmp_path / "out")
    assert [row["id"] for row in backend.direct] == ["one", "one", "one", "two"]
    assert len(predictions) == 2


def test_shared_mode_groups_by_state_and_prompt_condition(tmp_path, backend):
    questions = [dict(DECISION, id=f"q{index}", question=f"Question {index}?") for index in range(2)]
    rows = [variant for question in questions for variant in expand(question, langs=("en", "it"))]
    runner.run(rows, tmp_path / "out", make_args(mode="shared"), write_rows(tmp_path, rows))
    assert backend.shared == [["q0#en.json.r0", "q1#en.json.r0"], ["q0#it.json.r0", "q1#it.json.r0"]]
    predictions, _ = read_output(tmp_path / "out")
    assert [prediction["decision_id"] for prediction in predictions] == ["q0", "q1", "q0", "q1"]


def test_failed_load_still_leaves_a_manifest(tmp_path, monkeypatch):
    def broken(args):
        raise RuntimeError("Checkpoint did not load completely")

    monkeypatch.setattr(runner, "load_backend", broken)
    rows = [DECISION]
    with pytest.raises(RuntimeError):
        runner.run(rows, tmp_path / "out", make_args(), write_rows(tmp_path, rows))
    manifest = json.loads((tmp_path / "out" / "fake_variants_20261022-2000.manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "Checkpoint did not load completely" in manifest["error"]


def test_cli_refuses_a_run_id_that_already_has_files(tmp_path, monkeypatch, capsys):
    source = write_rows(tmp_path, [DECISION])
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "fake_variants_20261022-2000.manifest.json").write_text("{}")
    monkeypatch.setattr(sys, "argv", ["cs1-run", str(source), "--output", str(tmp_path / "out"), "--mode", "direct",
                                      "--model", "fake", "--revision", "a" * 40, "--run-id", "fake_variants_20261022-2000"])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert "already has files" in capsys.readouterr().err


def test_cli_refuses_model_ids_with_an_underscore(tmp_path, monkeypatch, capsys):
    source = write_rows(tmp_path, [DECISION])
    monkeypatch.setattr(sys, "argv", ["cs1-run", str(source), "--output", str(tmp_path / "out"), "--mode", "direct",
                                      "--model", "fake", "--model-id", "my_model", "--revision", "a" * 40])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert "contain no underscore" in capsys.readouterr().err


def test_progress_prints_one_line_per_interval_with_rate_and_eta():
    now = [0.0]
    lines = []
    progress = ui.Progress(total=100, every=10, clock=lambda: now[0],
                           stream=SimpleNamespace(write=lines.append, flush=lambda: None))
    now[0] = 5
    progress.advance([{}] * 10)
    assert lines == []  # Interval not reached yet.
    now[0] = 10
    progress.advance([{}] * 9 + [{"error": "x"}])
    assert lines[0] == "  ▸ 20/100 (20%), 2.0 rows/s, ETA 0:00:40, 1 error"


def test_run_reports_progress_on_stderr(tmp_path, backend, capsys):
    rows = [DECISION]
    runner.run(rows, tmp_path / "out", make_args(), write_rows(tmp_path, rows))
    err = capsys.readouterr().err
    assert err.startswith("  · loading weights (torch, direct)")  # Banner and title come from main().
    assert "  ✓ scored 1 row in" in err and "0 errors" in err and "  ✓ done in" in err


def test_system_role_in_the_input_is_ignored_when_the_template_supports_it(tmp_path, backend):
    rows = [dict(DECISION, id="old", system_role="merge")]  # e.g. from an older variants file
    runner.run(rows, tmp_path / "out", make_args(), write_rows(tmp_path, rows))
    assert backend.direct[0]["system_role"] == "native"


def test_allowed_remote_code_names_the_files_it_will_run(tmp_path, backend, capsys):
    model = tmp_path / "Spark-X2.5-4B"
    model.mkdir()
    (model / "modeling_spark.py").write_text("")
    (model / "configuration_spark.py").write_text("")
    rows = [DECISION]
    runner.run(rows, tmp_path / "out", make_args(model=str(model), allow_remote_code=True), write_rows(tmp_path, rows))
    assert "remote code allowed: running configuration_spark.py, modeling_spark.py" in capsys.readouterr().err


class Terminal(io.StringIO):
    def isatty(self):
        return True


def test_terminals_get_a_live_bar_and_a_final_summary_line():
    stream = Terminal()
    progress = ui.Progress(total=4, every=10, stream=stream)
    progress.start()
    assert progress.bar is not None
    progress.advance([{}, {"error": "x"}])
    progress.say("note above the bar")
    progress.advance([{}, {}])
    progress.close()
    output = stream.getvalue()
    assert progress.bar is None and "note above the bar" in output
    assert progress.failed == 1


def test_nested_runs_draw_their_bar_even_without_a_terminal(monkeypatch):
    monkeypatch.setenv("CS1_NESTED", "1")
    progress = ui.Progress(total=2, every=10, stream=io.StringIO())
    progress.start()
    assert progress.bar is not None  # The parent shows it on its status line.
    progress.close()


def test_on_a_terminal_main_follows_the_same_run_in_a_child_and_summarises_it(tmp_path, monkeypatch, capsys):
    source, output = write_rows(tmp_path, [DECISION]), tmp_path / "out"
    calls = []

    def follow(command, stream, first=""):
        calls.append(command)
        output.mkdir()
        run_id = command[command.index("--run-id") + 1]
        (output / f"{run_id}.manifest.json").write_text(json.dumps({
            "status": "completed", "backend": "torch", "model_class": "Qwen3ForCausalLM", "device": "cuda:0",
            "load_seconds": 81.1, "scoring_seconds": 2.0, "rows_written": 1, "rows_failed": 0, "wall_seconds": 90.0,
            "system_role": "native"}))
        return 0

    monkeypatch.delenv("CS1_NESTED", raising=False)
    monkeypatch.setattr(runner.ui, "is_terminal", lambda stream: True)
    monkeypatch.setattr(runner.ui, "follow", follow)
    monkeypatch.setattr(sys, "argv", ["cs1 run", str(source), "--output", str(output), "--mode", "direct",
                                      "--model", "fake", "--revision", "a" * 40])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 0
    arguments = calls[0][4:]
    assert arguments[:-2] == sys.argv[1:] and arguments[-2] == "--run-id"  # Same arguments, plus the run ID.
    assert arguments[-1].startswith("fake_variants_")
    err = capsys.readouterr().err
    assert "CS1-BENCH v" in err and "Starting test for fake (torch, direct, 1 row)" in err
    assert "loaded Qwen3ForCausalLM in 0:01:21 on cuda:0" in err
    assert "scored 1 row in 2.0s (0.5 rows/s), 0 errors, system role native" in err
    assert f"done in 0:01:30, {output}" in err
