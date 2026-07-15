import unittest
from contextlib import nullcontext

from eval_prompts import completion_metrics, generate_completion, parse_judge_response
from train_qlora import USER_INSTRUCTION, format_chat


class FakeTokenizer:
    def __init__(self):
        self.messages = None
        self.options = None

    def apply_chat_template(self, messages, **options):
        self.messages = messages
        self.options = options
        return "rendered chat"


class FakeTensor:
    def __init__(self, length):
        self.shape = (1, length)
        self.device = None

    def to(self, device):
        self.device = device
        return self


class FakeOutputIds:
    def __init__(self):
        self.index = None

    def __getitem__(self, index):
        self.index = index
        return "completion token ids"


class FakeGenerationTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self, encoded):
        self.encoded = encoded
        self.options = None
        self.decoded = None

    def apply_chat_template(self, messages, **options):
        self.options = options
        return self.encoded

    def decode(self, token_ids, **options):
        self.decoded = (token_ids, options)
        return " generated completion "


class FakeModel:
    device = "test-device"

    def __init__(self):
        self.output_ids = FakeOutputIds()
        self.generation_options = None

    def generate(self, **options):
        self.generation_options = options
        return self.output_ids


class FakeTorch:
    @staticmethod
    def inference_mode():
        return nullcontext()


class ChatFormattingTests(unittest.TestCase):
    def test_formats_brief_as_user_and_prompt_as_assistant(self):
        tokenizer = FakeTokenizer()
        row = {"brief": "A courier runs.", "prompt": "SUBJECT — Courier."}

        rendered = format_chat(tokenizer, row)

        self.assertEqual(rendered, "rendered chat")
        self.assertEqual(
            tokenizer.messages,
            [
                {
                    "role": "user",
                    "content": f"{USER_INSTRUCTION}\n\nA courier runs.",
                },
                {"role": "assistant", "content": "SUBJECT — Courier."},
            ],
        )
        self.assertEqual(
            tokenizer.options,
            {"tokenize": False, "add_generation_prompt": False},
        )


class JudgeParsingTests(unittest.TestCase):
    def test_parses_valid_response(self):
        response = (
            '{"skeleton": 3, "shot_grammar": 2.5, "constraints": 2, '
            '"specificity": 1.5, "total": 9}'
        )

        self.assertEqual(
            parse_judge_response(response),
            {
                "skeleton": 3.0,
                "shot_grammar": 2.5,
                "constraints": 2.0,
                "specificity": 1.5,
                "total": 9.0,
            },
        )

    def test_rejects_malformed_response(self):
        with self.assertRaisesRegex(ValueError, "valid judge JSON"):
            parse_judge_response("not JSON")


class CompletionMetricTests(unittest.TestCase):
    def test_wires_block_header_validator(self):
        valid = """SUBJECT — A courier.
LOCATION — A station.
ACTION — SHOT 1 (0:00–0:15) — Running.
CAMERA — Tracking.
STYLE — Blue 60%, gray 30%, red 10%.
CONSTRAINTS — 16:9.
"""

        self.assertEqual(completion_metrics(valid), {"block_header_valid": True})
        self.assertEqual(
            completion_metrics(valid.replace("CAMERA —", "FRAMING —")),
            {"block_header_valid": False},
        )


class GenerationTests(unittest.TestCase):
    def assert_generates_from(self, encoded, input_ids):
        tokenizer = FakeGenerationTokenizer(encoded)
        model = FakeModel()

        completion = generate_completion(model, tokenizer, FakeTorch, "A courier runs.")

        self.assertEqual(completion, "generated completion")
        self.assertEqual(
            tokenizer.options,
            {
                "tokenize": True,
                "add_generation_prompt": True,
                "return_tensors": "pt",
                "return_dict": True,
            },
        )
        self.assertEqual(input_ids.device, model.device)
        self.assertEqual(
            model.generation_options,
            {
                "input_ids": input_ids,
                "max_new_tokens": 1536,
                "do_sample": False,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            },
        )
        self.assertEqual(model.output_ids.index, (0, slice(input_ids.shape[-1], None)))
        self.assertEqual(
            tokenizer.decoded,
            ("completion token ids", {"skip_special_tokens": True}),
        )

    def test_generates_when_chat_template_returns_tensor(self):
        input_ids = FakeTensor(length=4)

        self.assert_generates_from(input_ids, input_ids)

    def test_generates_when_chat_template_returns_dict(self):
        input_ids = FakeTensor(length=6)

        self.assert_generates_from({"input_ids": input_ids}, input_ids)


if __name__ == "__main__":
    unittest.main()
