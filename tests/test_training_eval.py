import unittest

from eval_prompts import completion_metrics, parse_judge_response
from train_qlora import USER_INSTRUCTION, format_chat


class FakeTokenizer:
    def __init__(self):
        self.messages = None
        self.options = None

    def apply_chat_template(self, messages, **options):
        self.messages = messages
        self.options = options
        return "rendered chat"


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


if __name__ == "__main__":
    unittest.main()
