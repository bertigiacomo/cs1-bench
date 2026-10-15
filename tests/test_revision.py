import pytest

from cs1_bench.core import local_revision, resolve_model_source, resolve_revision

COMMIT_A, COMMIT_B = "a" * 40, "b" * 40


def write_metadata(folder, name, commit):
    path = folder / ".cache" / "huggingface" / "download" / f"{name}.metadata"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{commit}\n{'e' * 40}\n1700000000.0\n")


def test_hf_download_metadata_gives_the_revision(tmp_path):
    write_metadata(tmp_path, "config.json", COMMIT_A)
    write_metadata(tmp_path, "model.safetensors", COMMIT_A)
    assert resolve_revision(str(tmp_path), None) == (COMMIT_A, "hf-download-metadata")


def test_disagreeing_metadata_is_an_error(tmp_path):
    write_metadata(tmp_path, "config.json", COMMIT_A)
    write_metadata(tmp_path, "model.safetensors", COMMIT_B)
    with pytest.raises(ValueError, match="disagree"):
        local_revision(tmp_path)


def test_revision_file_is_the_fallback_and_must_be_a_commit(tmp_path):
    (tmp_path / "REVISION").write_text(COMMIT_B + "\n")
    assert local_revision(tmp_path) == (COMMIT_B, "revision-file")
    (tmp_path / "REVISION").write_text("main\n")
    with pytest.raises(ValueError, match="40-character"):
        local_revision(tmp_path)


def test_explicit_revision_wins_and_missing_ones_explain_what_to_do(tmp_path):
    write_metadata(tmp_path, "config.json", COMMIT_A)
    assert resolve_revision(str(tmp_path), "manual") == ("manual", "argument")
    empty = tmp_path / "empty-model"
    empty.mkdir()
    with pytest.raises(ValueError, match="REVISION file"):
        resolve_revision(str(empty), None)
    with pytest.raises(ValueError, match="Remote models"):
        resolve_revision("Qwen/Qwen3.5-4B", None)


def test_bare_names_resolve_under_cs1_models_dir(tmp_path, monkeypatch):
    (tmp_path / "Qwen3.5-4B").mkdir()
    monkeypatch.setenv("CS1_MODELS_DIR", str(tmp_path))
    assert resolve_model_source("Qwen3.5-4B") == str(tmp_path / "Qwen3.5-4B")
    assert resolve_model_source("Qwen/Qwen3.5-4B") == "Qwen/Qwen3.5-4B"
