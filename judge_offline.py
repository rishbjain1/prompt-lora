"""Judge existing base and adapter evaluation outputs without generating prompts."""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import httpx

from eval_prompts import API_URL, SCORE_KEYS, summarize

# The judge tends to echo the rubric's dimension names and to append prose
# after the JSON block, so extraction has to be tolerant of both.
KEY_ALIASES = {
    "skeleton_compliance": "skeleton",
    "constraint_adherence": "constraints",
}


def extract_judge_scores(text: str) -> dict[str, float]:
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        raise ValueError("Judge did not return valid judge JSON")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError("Judge did not return valid judge JSON") from exc
    normalized = {KEY_ALIASES.get(key, key): value for key, value in payload.items()}
    try:
        return {key: float(normalized[key]) for key in SCORE_KEYS}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Judge did not return valid judge JSON") from exc


OUTPUT_DIR = Path("eval/out")
RUBRIC_PATH = Path("eval/rubric.md")
DEFAULT_JUDGE_MODEL = "claude-haiku-4-5-20251001"


def judge_completion(
    client: httpx.Client,
    api_key: str,
    model: str,
    rubric: str,
    brief: str,
    reference_prompt: str,
    completion: str,
) -> dict[str, float]:
    """Score one existing completion, using its reference prompt as gold context."""
    response = client.post(
        API_URL,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 500,
            "temperature": 0,
            "system": (
                "Score the generated prompt against the supplied brief and rubric. The gold "
                "reference prompt is context for expected quality, not the item being scored. "
                "Return JSON only with numeric keys skeleton, shot_grammar, constraints, "
                "specificity, total. Map rubric skeleton_compliance to skeleton and "
                "constraint_adherence to constraints."
            ),
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"RUBRIC:\n{rubric}\n\nBRIEF:\n{brief}\n\n"
                        f"GOLD REFERENCE PROMPT:\n{reference_prompt}\n\n"
                        f"COMPLETION:\n{completion}"
                    ),
                }
            ],
        },
    )
    response.raise_for_status()
    payload = response.json()
    text_parts = [
        block["text"]
        for block in payload.get("content", [])
        if block.get("type") == "text" and block.get("text")
    ]
    if not text_parts:
        raise ValueError("Anthropic response contained no text")
    return extract_judge_scores("\n".join(text_parts))


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError(f"Invalid evaluation output: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _completed_judge(row: Mapping[str, Any]) -> bool:
    judge = row.get("judge")
    return isinstance(judge, Mapping) and all(judge.get(key) is not None for key in SCORE_KEYS)


def _same_result(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    fields = ("brief", "reference_prompt", "completion", "block_header_valid")
    return all(left.get(field) == right.get(field) for field in fields)


def judge_existing_output(
    source_path: Path,
    output_path: Path,
    judge_model: str,
    judge_fn: Callable[[str, str, str], dict[str, float]],
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Judge one evaluation file, reusing matching completed rows from prior output."""
    source = _load_json(source_path)
    existing = _load_json(output_path) if output_path.exists() else None
    existing_rows = existing["results"] if existing else []

    results: list[dict[str, Any]] = []
    for index, source_row in enumerate(source["results"]):
        result = dict(source_row)
        if index < len(existing_rows):
            old_row = existing_rows[index]
            if _same_result(result, old_row) and _completed_judge(old_row):
                result["judge"] = old_row["judge"]
                result["judge_error"] = old_row.get("judge_error")
        results.append(result)

    pending = [index for index, result in enumerate(results) if not _completed_judge(result)]
    output = dict(source)
    output["judge_model"] = judge_model
    output["results"] = results

    for pending_index, result_index in enumerate(pending):
        result = results[result_index]
        try:
            result["judge"] = judge_fn(
                result["brief"],
                result["reference_prompt"],
                result["completion"],
            )
            result["judge_error"] = None
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            result["judge"] = {key: None for key in SCORE_KEYS}
            result["judge_error"] = str(exc)
        output["means"] = summarize(results, judged=True)
        _write_json(output_path, output)
        print(f"Judged {result_index + 1}/{len(results)} from {source_path}")
        if pending_index < len(pending) - 1:
            sleep_fn(0.3)

    output["means"] = summarize(results, judged=True)
    _write_json(output_path, output)
    return output


def print_comparison(base: Mapping[str, Any], adapter: Mapping[str, Any]) -> None:
    """Print base-versus-adapter mean scores."""
    metrics = ("block_header_valid", *SCORE_KEYS)
    print("\nmetric                 base     adapter")
    print("--------------------  -------  -------")
    for metric in metrics:
        base_value = base["means"].get(metric)
        adapter_value = adapter["means"].get(metric)
        base_text = "-" if base_value is None else f"{base_value:.3f}"
        adapter_text = "-" if adapter_value is None else f"{adapter_value:.3f}"
        print(f"{metric:<20}  {base_text:>7}  {adapter_text:>7}")


def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CRAG_LLM_API_KEY")
    if not api_key:
        raise SystemExit("Set ANTHROPIC_API_KEY or CRAG_LLM_API_KEY")
    judge_model = os.environ.get("JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
    rubric = RUBRIC_PATH.read_text(encoding="utf-8")

    client = httpx.Client(timeout=120.0)
    try:
        judge_fn = lambda brief, reference, completion: judge_completion(
            client,
            api_key,
            judge_model,
            rubric,
            brief,
            reference,
            completion,
        )
        base = judge_existing_output(
            OUTPUT_DIR / "base.json",
            OUTPUT_DIR / "base_judged.json",
            judge_model,
            judge_fn,
        )
        time.sleep(0.3)
        adapter = judge_existing_output(
            OUTPUT_DIR / "adapter.json",
            OUTPUT_DIR / "adapter_judged.json",
            judge_model,
            judge_fn,
        )
    finally:
        client.close()

    print_comparison(base, adapter)


if __name__ == "__main__":
    main()
