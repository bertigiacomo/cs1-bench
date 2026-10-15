import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from cs1_bench.direct import score  # noqa: E402

ROW = {
    "id": "x",
    "state": "owned state",
    "question": "Which answer follows?",
    "options": [{"id": "yes", "description": "Yes."}, {"id": "no", "description": "No."}],
    "prompt_lang": "it",
    "prompt_format": "text",
}


class Tokenizer:
    """Byte tokenizer: "A" and "B" are single tokens 65 and 66."""

    def apply_chat_template(self, turns, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return "\n".join(turn["content"] for turn in turns) + "\nASSISTANT\n"

    def encode(self, text, add_special_tokens=False):
        return list(text.encode())

    def decode(self, ids):
        return bytes(ids).decode()


class FakeModel:
    """Puts logit 2 on "A", 1 on "B", and 0 on the other 254 byte tokens."""

    def parameters(self):
        yield SimpleNamespace(device=torch.device("cpu"))

    def forward(self, input_ids=None, attention_mask=None, use_cache=False, return_dict=True, logits_to_keep=1):
        logits = torch.zeros((1, 1, 256))
        logits[0, 0, ord("A")], logits[0, 0, ord("B")] = 2.0, 1.0
        return SimpleNamespace(logits=logits)

    __call__ = forward


def test_direct_reports_condition_and_mass_on_answer_letters():
    result = score(FakeModel(), Tokenizer(), ROW, {"source": "fake"})
    letters = math.exp(2) + math.exp(1)
    assert result["allowed_token_mass"] == pytest.approx(letters / (letters + 254))
    assert result["full_vocab_argmax_id"] == ord("A")
    assert result["probabilities"][0] > result["probabilities"][1]
    assert (result["prompt_lang"], result["prompt_format"], result["system_role"]) == ("it", "text", "native")
