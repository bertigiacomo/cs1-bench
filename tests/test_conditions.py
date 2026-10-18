import json
import sys

import pytest

from cs1_bench import conditions
from cs1_bench.core import direct_messages

DECISION = {
    "id": "d1",
    "state": "Customer cannot log in.",
    "question": "Which queue?",
    "options": [
        {"id": "account", "description": "Account access."},
        {"id": "billing", "description": "Billing."},
        {"id": "sales", "description": "Sales."},
    ],
    "answer": "account",
    "lang": "en",
}


def test_cyclic_rotation_puts_every_option_in_every_position_once():
    variants = conditions.expand(DECISION, rotations="cyclic")
    orders = [[option["id"] for option in variant["options"]] for variant in variants]
    assert orders == [["account", "billing", "sales"], ["billing", "sales", "account"], ["sales", "account", "billing"]]
    assert {order.index("account") for order in orders} == {0, 1, 2}
    assert all(variant["answer"] == "account" for variant in variants)


def test_grid_is_the_full_product_with_unique_ids():
    variants = conditions.expand(DECISION, langs=("en", "it"), formats=("json", "text"), rotations="cyclic")
    assert len(variants) == 2 * 2 * 3
    assert len({variant["id"] for variant in variants}) == len(variants)
    assert {variant["base_id"] for variant in variants} == {"d1"}
    assert "d1#it.text.r2" in {variant["id"] for variant in variants}


def test_variant_keeps_data_fields_and_records_its_condition():
    [variant] = conditions.expand(DECISION, langs=("it",))
    assert variant["lang"] == "en"  # Data language stays independent of the prompt language.
    assert {field: variant[field] for field in conditions.VARIANT_FIELDS} == {
        "base_id": "d1", "rotation": 0, "prompt_lang": "it", "prompt_format": "json",
    }
    assert "system_role" not in variant  # The runner picks it from the model's chat template.


def test_variant_bookkeeping_never_reaches_the_prompt():
    [variant] = conditions.expand(DECISION, rotations="none")
    rendered = json.dumps(direct_messages(variant))
    assert "base_id" not in rendered and "rotation" not in rendered and "d1" not in rendered


def test_unknown_condition_is_rejected():
    with pytest.raises(ValueError, match="prompt_lang"):
        conditions.expand(DECISION, langs=("fr",))


def test_cli_writes_variants_and_rejects_bad_lists(tmp_path, monkeypatch, capsys):
    source, output = tmp_path / "decisions.jsonl", tmp_path / "variants.jsonl"
    source.write_text(json.dumps(DECISION) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["cs1-variants", str(source), "--output", str(output),
                                      "--langs", "en,it", "--rotations", "cyclic"])
    conditions.main()
    assert len(output.read_text(encoding="utf-8").splitlines()) == 6
    monkeypatch.setattr(sys, "argv", ["cs1-variants", str(source), "--output", str(tmp_path / "other.jsonl"),
                                      "--formats", "json,yaml"])
    with pytest.raises(SystemExit) as error:
        conditions.main()
    assert error.value.code == 2
    assert "comma-separated subset" in capsys.readouterr().err


def test_reverse_adds_only_the_reversed_order():
    original, reversed_ = conditions.expand(DECISION, rotations="reverse")
    assert [option["id"] for option in reversed_["options"]] == ["sales", "billing", "account"]
    assert (original["rotation"], reversed_["rotation"]) == (0, "reverse")
    assert reversed_["id"] == "d1#en.json.rev" and reversed_["answer"] == "account"
