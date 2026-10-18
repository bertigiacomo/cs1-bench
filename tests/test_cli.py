"""Shared model options and loader: argument checks and backend selection, with mocks (no model, no GPU)."""

import argparse
from unittest.mock import Mock

import pytest

from cs1_bench import cli


def parse(*argv):
    """Parse and check model arguments the way cs1 run and cs1 check do."""
    parser = argparse.ArgumentParser()
    cli.add_model_arguments(parser)
    args = parser.parse_args(["--model", "test/model", "--revision", "a" * 40, *argv])
    cli.check_model_arguments(parser, args)
    return args


@pytest.mark.parametrize("flags", [[], ["--device", "auto"], ["--device", "cuda"]])
def test_reranker_always_loads_on_cuda(flags):
    assert parse("--mode", "reranker", *flags).device == "cuda"


@pytest.mark.parametrize("device", ["mps", "cpu"])
def test_reranker_rejects_non_cuda_devices(capsys, device):
    with pytest.raises(SystemExit) as error:
        parse("--mode", "reranker", "--device", device)
    assert error.value.code == 2
    assert "Reranker mode requires CUDA" in capsys.readouterr().err


@pytest.mark.parametrize("flags,expected", [
    (["--device", "cpu", "--dtype", "float32"], ("cpu", "float32", False)),
    (["--allow-remote-code"], ("auto", "bfloat16", True)),
])
def test_torch_loader_receives_device_dtype_and_remote_code(monkeypatch, flags, expected):
    loader = Mock(return_value=("model", "tokenizer", {}))
    monkeypatch.setattr(cli, "load_causal_model", loader)
    _, _, _, scorers = cli.load_backend(parse("--mode", "direct", *flags))
    device, dtype, remote = expected
    loader.assert_called_once_with("test/model", "a" * 40, device, dtype, trust_remote_code=remote)
    assert scorers.direct is cli.direct_score and scorers.reranker is cli.reranker_score
