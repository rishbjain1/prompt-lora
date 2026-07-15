import json
import tempfile
import unittest
from pathlib import Path

from judge_offline import judge_existing_output, judge_completion


SCORES = {
    "skeleton": 3.0,
    "shot_grammar": 2.0,
    "constraints": 2.5,
    "specificity": 1.5,
    "total": 9.0,
}


class FakeResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {"content": [{"type": "text", "text": json.dumps(SCORES)}]}


class FakeClient:
    def __init__(self):
        self.request = None

    def post(self, *args, **kwargs):
        self.request = (args, kwargs)
        return FakeResponse()


def evaluation(rows):
    return {
        "name": "base",
        "base_model": "base-model",
        "adapter": None,
        "judge_model": None,
        "count": len(rows),
        "means": {"block_header_valid": 0.0},
        "results": rows,
    }


def row(number, valid=True):
    return {
        "brief": f"brief {number}",
        "reference_prompt": f"gold {number}",
        "completion": f"completion {number}",
        "block_header_valid": valid,
    }


class JudgeCompletionTests(unittest.TestCase):
    def test_includes_reference_prompt_as_gold_example(self):
        client = FakeClient()

        scores = judge_completion(
            client,
            "api-key",
            "judge-model",
            "rubric text",
            "brief text",
            "gold prompt",
            "generated prompt",
        )

        self.assertEqual(scores, SCORES)
        body = client.request[1]["json"]
        self.assertIn("GOLD REFERENCE PROMPT:\ngold prompt", body["messages"][0]["content"])
        self.assertIn("COMPLETION:\ngenerated prompt", body["messages"][0]["content"])


class JudgeExistingOutputTests(unittest.TestCase):
    def test_judges_rows_updates_means_and_sleeps_between_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "base.json"
            output = Path(directory) / "base_judged.json"
            source.write_text(json.dumps(evaluation([row(1), row(2, valid=False)])))
            calls = []
            sleeps = []

            judge_existing_output(
                source,
                output,
                "judge-model",
                lambda brief, reference, completion: calls.append(
                    (brief, reference, completion)
                )
                or SCORES,
                sleeps.append,
            )

            judged = json.loads(output.read_text())
            self.assertEqual(len(calls), 2)
            self.assertEqual(sleeps, [0.3])
            self.assertEqual(judged["results"][0]["judge"], SCORES)
            self.assertEqual(judged["judge_model"], "judge-model")
            self.assertEqual(
                judged["means"],
                {"block_header_valid": 0.5, **SCORES},
            )

    def test_resumes_by_skipping_matching_already_judged_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "base.json"
            output = Path(directory) / "base_judged.json"
            source.write_text(json.dumps(evaluation([row(1), row(2)])))
            existing = evaluation([row(1), row(2)])
            existing["results"][0]["judge"] = SCORES
            existing["results"][0]["judge_error"] = None
            output.write_text(json.dumps(existing))
            calls = []

            judge_existing_output(
                source,
                output,
                "judge-model",
                lambda brief, reference, completion: calls.append(brief) or SCORES,
                lambda _: None,
            )

            self.assertEqual(calls, ["brief 2"])
            judged = json.loads(output.read_text())
            self.assertEqual(judged["results"][0]["judge"], SCORES)
            self.assertEqual(judged["results"][1]["judge"], SCORES)


if __name__ == "__main__":
    unittest.main()
