"""CLI routing tests with mocked backends: no model downloads or GPU needed."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from cs1_bench import cli


@pytest.fixture
def run_cli(tmp_path, monkeypatch):
    rows = [
        {"id": key, "state": "shared evidence", "question": "Which answer follows?",
         "options": [{"id": "yes", "description": "Yes."}, {"id": "no", "description": "No."}]}
        for key in ("first", "second")
    ]
    source = tmp_path / "input.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    output = tmp_path / "nested" / "output.jsonl"

    def run(mode="direct", *flags):
        monkeypatch.setattr(sys, "argv", [
            "cs1-score", "--mode", mode, "--model", "test/model",
            "--revision", "a" * 40, "--input", str(source), "--output", str(output),
            "--max-tokens", "128", *flags,
        ])
        cli.main()

    return SimpleNamespace(run=run, rows=rows, output=output)


@pytest.fixture
def backends(monkeypatch):
    loaded = (object(), object(), {"revision": "a" * 40})
    results = [{"id": "first", "probabilities": [0.25, 0.75]},
               {"id": "second", "probabilities": [0.8, 0.2]}]
    timing = {"total_seconds": 0.125}
    torch = SimpleNamespace(
        load_causal_model=Mock(return_value=loaded),
        direct_score=Mock(side_effect=results),
        score_shared=Mock(return_value=(results, timing)),
        SerialPrefixScorer=Mock(return_value=SimpleNamespace(score=Mock(side_effect=results))),
        reranker_score=Mock(side_effect=results),
    )
    for name, mock in vars(torch).items():
        monkeypatch.setattr(cli, name, mock)
    return SimpleNamespace(torch=torch, loaded=loaded, results=results, timing=timing)


@pytest.mark.parametrize("mode", ["direct", "serial", "shared"])
def test_torch_routes_to_matching_scorer(run_cli, backends, mode):
    run_cli.run(mode)
    torch = backends.torch
    model, tokenizer, metadata = backends.loaded
    torch.load_causal_model.assert_called_once()
    if mode == "direct":
        assert torch.direct_score.call_args_list == [call(model, tokenizer, row, metadata, 128) for row in run_cli.rows]
    elif mode == "serial":
        assert torch.SerialPrefixScorer.return_value.score.call_args_list == [call(row) for row in run_cli.rows]
    else:
        torch.score_shared.assert_called_once_with(model, tokenizer, run_cli.rows, metadata, 128)
    expected = backends.results if mode != "shared" else [
        {**result, "shared_timing": backends.timing} for result in backends.results
    ]
    assert [json.loads(line) for line in run_cli.output.read_text().splitlines()] == expected


@pytest.mark.parametrize("flags", [[], ["--device", "auto"], ["--device", "cuda"]])
def test_torch_reranker_always_loads_cuda(run_cli, backends, flags):
    run_cli.run("reranker", *flags)
    backends.torch.load_causal_model.assert_called_once_with("test/model", "a" * 40, "cuda", "bfloat16")
    model, tokenizer, metadata = backends.loaded
    assert backends.torch.reranker_score.call_args_list == [
        call(model, tokenizer, row, metadata, 128) for row in run_cli.rows
    ]


@pytest.mark.parametrize("device", ["mps", "cpu"])
def test_torch_reranker_rejects_non_cuda_device_before_loading(run_cli, backends, capsys, device):
    run_cli.run("reranker")
    run_cli.output.unlink()
    with pytest.raises(SystemExit) as error:
        run_cli.run("reranker", "--device", device)
    assert error.value.code == 2
    assert "Reranker mode requires CUDA" in capsys.readouterr().err
    assert backends.torch.reranker_score.call_count == 2


def test_torch_cpu_selection_reaches_loader(run_cli, backends):
    run_cli.run("direct", "--device", "cpu", "--dtype", "float32")
    backends.torch.load_causal_model.assert_called_once_with(
        "test/model", "a" * 40, "cpu", "float32")


def test_existing_output_is_not_overwritten(run_cli, backends, capsys):
    original = b"existing benchmark evidence\n"
    run_cli.output.parent.mkdir(parents=True)
    run_cli.output.write_bytes(original)
    with pytest.raises(SystemExit) as error:
        run_cli.run("direct")
    assert error.value.code == 2
    assert "Output must be new" in capsys.readouterr().err
    assert run_cli.output.read_bytes() == original
    backends.torch.load_causal_model.assert_not_called()
