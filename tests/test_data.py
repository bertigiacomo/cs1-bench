import json
import sys

import pytest

from cs1_bench import data
from cs1_bench.core import direct_messages


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_csv_with_json_options_column(tmp_path):
    source = write(tmp_path, "in.csv",
                   'id,state,question,options,answer\n'
                   'r1,"Ticket text, with comma",Which queue?,'
                   '"[{""id"": ""billing"", ""description"": ""Billing""}, '
                   '{""id"": ""sales"", ""description"": ""Sales""}]",billing\n')
    assert data.import_file(source, tmp_path / "out.jsonl") == 1
    [row] = read_jsonl(tmp_path / "out.jsonl")
    assert row["state"] == "Ticket text, with comma"
    assert [option["id"] for option in row["options"]] == ["billing", "sales"]
    assert row["answer"] == "billing"


def test_wide_option_columns_semicolon_and_italian_text(tmp_path):
    source = write(tmp_path, "in.csv",
                   "state;question;option_si;option_no;option_forse;answer\n"
                   "Il deploy è riuscito.;È andato a buon fine?;Sì;No;;si\n")
    data.import_file(source, tmp_path / "out.jsonl", lang="it")
    [row] = read_jsonl(tmp_path / "out.jsonl")
    # The empty option_forse cell means this row has only two options.
    assert row["options"] == [{"id": "si", "description": "Sì"}, {"id": "no", "description": "No"}]
    assert row["lang"] == "it"
    assert "è riuscito" in (tmp_path / "out.jsonl").read_text(encoding="utf-8")


def test_column_mapping_and_extra_fields_go_to_meta(tmp_path):
    source = write(tmp_path, "in.jsonl", json.dumps({
        "evidence": "Health checks passed.", "criterion": "Did it succeed?",
        "options": ["yes", "no"], "gold": "yes", "source": "authored",
    }) + "\n")
    data.import_file(source, tmp_path / "out.jsonl",
                     mapping={"state": "evidence", "question": "criterion", "answer": "gold"})
    [row] = read_jsonl(tmp_path / "out.jsonl")
    assert row["state"] == "Health checks passed."
    assert row["options"][0] == {"id": "yes", "description": "yes"}
    assert row["meta"] == {"source": "authored"}


def test_generated_ids_are_stable_across_row_order(tmp_path):
    first = {"state": "s1", "question": "q", "options": ["a", "b"]}
    second = {"state": "s2", "question": "q", "options": ["a", "b"]}
    data.import_file(write(tmp_path, "one.json", json.dumps([first, second])), tmp_path / "one.jsonl")
    data.import_file(write(tmp_path, "two.json", json.dumps([second, first])), tmp_path / "two.jsonl")
    one, two = read_jsonl(tmp_path / "one.jsonl"), read_jsonl(tmp_path / "two.jsonl")
    assert one[0]["id"] == two[1]["id"] and one[1]["id"] == two[0]["id"]
    assert one[0]["id"] != one[1]["id"]


def test_answer_must_be_an_option_id(tmp_path):
    source = write(tmp_path, "in.json", json.dumps([{"state": "s", "question": "q",
                                                     "options": ["a", "b"], "answer": "c"}]))
    with pytest.raises(ValueError, match="record 1: answer 'c'"):
        data.import_file(source, tmp_path / "out.jsonl")
    assert not (tmp_path / "out.jsonl").exists()


def test_duplicate_ids_are_rejected(tmp_path):
    row = {"id": "same", "state": "s", "question": "q", "options": ["a", "b"]}
    with pytest.raises(ValueError, match="Duplicate decision IDs"):
        data.import_file(write(tmp_path, "in.json", json.dumps([row, row])), tmp_path / "out.jsonl")


def test_existing_output_is_not_overwritten(tmp_path):
    source = write(tmp_path, "in.json", json.dumps([{"state": "s", "question": "q", "options": ["a", "b"]}]))
    output = write(tmp_path, "out.jsonl", "existing\n")
    with pytest.raises(FileExistsError):
        data.import_file(source, output)
    assert output.read_text() == "existing\n"


def test_gold_answer_never_reaches_the_prompt(tmp_path):
    source = write(tmp_path, "in.json", json.dumps([{"state": "s", "question": "q",
                                                     "options": ["alpha", "beta"], "answer": "beta",
                                                     "lang": "en", "notes": "hidden"}]))
    data.import_file(source, tmp_path / "out.jsonl")
    [row] = read_jsonl(tmp_path / "out.jsonl")
    rendered = json.dumps(direct_messages(row))
    assert "answer" not in rendered and "hidden" not in rendered and '"lang"' not in rendered


def test_cli_reports_record_errors(tmp_path, monkeypatch, capsys):
    source = write(tmp_path, "in.csv", "state,question,option_a\ns,q,only one\n")
    monkeypatch.setattr(sys, "argv", ["cs1-import", str(source), "--output", str(tmp_path / "out.jsonl")])
    with pytest.raises(SystemExit) as error:
        data.main()
    assert error.value.code == 2
    assert "in.csv record 1: options must contain 2-16 entries" in capsys.readouterr().err
