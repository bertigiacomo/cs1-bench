"""The common-format row: order labels, sets, ties and failed rows, without a model."""

import datetime

import pytest

from cs1_bench import output

ROW = {"id": "dom-1#en.text.native.reverse", "base_id": "dom-1", "rotation": "reverse", "prompt_format": "text",
       "options": [{"id": "gas", "description": "Gas."}, {"id": "energy", "description": "Power."}], "answer": "energy"}


@pytest.mark.parametrize("rotation,label", [(None, "dir0"), (0, "dir0"), (3, "dir3"), ("reverse", "rev0")])
def test_order_labels(rotation, label):
    assert output.order_label(rotation) == label


def test_unknown_rotation_is_refused():
    with pytest.raises(ValueError):
        output.order_label("r3")


def test_set_comes_from_the_row_or_the_id_prefix():
    assert output.decision_set({"set": "domain"}, "dom-1") == "domain"
    assert output.decision_set({}, "task-9") == "task"


def test_ties_go_to_the_first_option_shown():
    assert output.most_likely({"gas": 0.5, "energy": 0.5}, ["energy", "gas"]) == "energy"


def test_run_id_is_utc_and_refuses_underscores():
    started = datetime.datetime(2026, 10, 22, 22, 5, tzinfo=datetime.timezone(datetime.timedelta(hours=2)))
    assert output.run_id("gemma-4-E4B-it", "cosman-v1-base", started) == "gemma-4-E4B-it_cosman-v1-base_20261022-2005"
    with pytest.raises(ValueError):
        output.run_id("my_model", "d", started)


def test_failed_row_keeps_the_facts_and_has_no_prediction():
    record = output.common_row("r", ROW, {"error": "too long"})
    assert (record["decision_id"], record["set"], record["order"], record["option_ids"]) == (
        "dom-1", "dom", "rev0", ["gas", "energy"])
    assert (record["prediction"], record["probabilities"], record["error"]) == (None, None, "too long")


def test_scorer_options_must_match_the_order_shown():
    with pytest.raises(ValueError):
        output.common_row("r", ROW, {"option_ids": ["energy", "gas"], "probabilities": [0.9, 0.1],
                                     "option_logits": [1.0, 0.0]})
