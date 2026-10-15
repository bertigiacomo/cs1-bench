import pytest

from cs1_bench.core import PROMPTS
from cs1_bench.direct import encode_prompt
from cs1_bench.serial import _state_prefix

ROW = {
    "id": "x",
    "state": "owned state",
    "question": "Which answer follows?",
    "options": [{"id": "yes", "description": "Yes."}, {"id": "no", "description": "No."}],
}


class Tokenizer:
    def apply_chat_template(self, turns, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        assert tokenize is False and add_generation_prompt is True and enable_thinking is False
        return "HEADER\n" + "\n".join(turn["content"] for turn in turns) + "\nASSISTANT"

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(text.encode())

    def decode(self, ids):
        return bytes(ids).decode()


def test_state_prefix_stops_before_runtime_question_and_options():
    prefix = bytes(_state_prefix(Tokenizer(), ROW)).decode()
    assert "owned state" in prefix
    assert "prefix boundary placeholder" not in prefix
    assert '"options"' not in prefix


@pytest.mark.parametrize("lang", ["en", "it"])
@pytest.mark.parametrize("prompt_format", ["json", "text"])
@pytest.mark.parametrize("role", ["native", "merge"])
def test_state_prefix_matches_full_prompt_for_every_condition(lang, prompt_format, role):
    row = dict(ROW, prompt_lang=lang, prompt_format=prompt_format, system_role=role)
    prefix = _state_prefix(Tokenizer(), row)
    ids, _, _ = encode_prompt(Tokenizer(), row, 4096)
    assert ids[: len(prefix)] == prefix
    text = bytes(prefix).decode()
    assert PROMPTS[lang]["evidence"] in text and "owned state" in text
    assert ROW["question"] not in text
