"""Generate base/adapter completions and score them online or offline."""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
from statistics import mean
from typing import Any

import httpx

from distill import validate_structured_prompt
from train_qlora import DEFAULT_BASE_MODEL, USER_INSTRUCTION


API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_INPUT = Path("data/val_structured.jsonl")
DEFAULT_OUTPUT_DIR = Path("eval/out")
RUBRIC_PATH = Path("eval/rubric.md")
SCORE_KEYS = ("skeleton", "shot_grammar", "constraints", "specificity", "total")


def parse_judge_response(text: str) -> dict[str, float]:
    """Parse required numeric scores from judge JSON."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(cleaned)
        scores = {key: float(payload[key]) for key in SCORE_KEYS}
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("Judge did not return valid judge JSON") from exc
    return scores


def completion_metrics(completion: str) -> dict[str, bool]:
    """Compute free structural metrics for one completion."""
    return {"block_header_valid": validate_structured_prompt(completion)}


def load_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                rows.append({"brief": row["brief"], "prompt": row["prompt"]})
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"Invalid validation row in {path}:{line_number}") from exc
    return rows


def load_generator(base_model: str, adapter: Path | None):
    """Load inference model, using 4-bit weights when bitsandbytes is usable."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    use_cuda = torch.cuda.is_available()
    use_mps = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    dtype = (
        torch.bfloat16
        if use_cuda and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    model_kwargs: dict[str, Any] = {}
    if use_cuda:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            warnings.warn(
                "bitsandbytes unavailable; evaluating with unquantized weights.",
                stacklevel=2,
            )
            model_kwargs.update(torch_dtype=dtype, device_map="auto")
        else:
            model_kwargs.update(
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=dtype,
                ),
                device_map="auto",
            )
    elif use_mps:
        model_kwargs["torch_dtype"] = torch.float16
    else:
        model_kwargs["torch_dtype"] = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    if use_mps:
        model.to("mps")
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    return model, tokenizer, torch


def generate_completion(model: Any, tokenizer: Any, torch: Any, brief: str) -> str:
    messages = [
        {
            "role": "user",
            "content": f"{USER_INSTRUCTION}\n\n{brief}",
        }
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=1536,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0, input_ids.shape[-1] :], skip_special_tokens=True).strip()


def judge_completion(
    client: httpx.Client,
    api_key: str,
    model: str,
    rubric: str,
    brief: str,
    completion: str,
) -> dict[str, float]:
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
                "Score generated prompt against supplied brief and rubric. Return JSON only "
                "with numeric keys skeleton, shot_grammar, constraints, specificity, total. "
                "Map rubric skeleton_compliance to skeleton and constraint_adherence to constraints."
            ),
            "messages": [
                {
                    "role": "user",
                    "content": f"RUBRIC:\n{rubric}\n\nBRIEF:\n{brief}\n\nCOMPLETION:\n{completion}",
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
    return parse_judge_response("\n".join(text_parts))


def summarize(results: list[dict[str, Any]], judged: bool) -> dict[str, float | None]:
    summary: dict[str, float | None] = {
        "block_header_valid": mean(
            float(row["block_header_valid"]) for row in results
        )
        if results
        else None
    }
    if judged:
        for key in SCORE_KEYS:
            values = [
                row["judge"][key]
                for row in results
                if row.get("judge") and row["judge"].get(key) is not None
            ]
            summary[key] = mean(values) if values else None
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--name")
    parser.add_argument("--skip-judge", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_model = os.environ.get("BASE_MODEL", DEFAULT_BASE_MODEL)
    judge_model = os.environ.get("JUDGE_MODEL", "claude-haiku-4-5-20251001")
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CRAG_LLM_API_KEY")
    if not args.skip_judge and not api_key:
        raise SystemExit("Set ANTHROPIC_API_KEY or CRAG_LLM_API_KEY, or use --skip-judge")

    rows = load_rows(args.input)
    model, tokenizer, torch = load_generator(base_model, args.adapter)
    rubric = RUBRIC_PATH.read_text(encoding="utf-8") if not args.skip_judge else ""
    results: list[dict[str, Any]] = []
    client = httpx.Client(timeout=120.0) if not args.skip_judge else None
    try:
        for index, row in enumerate(rows, start=1):
            completion = generate_completion(model, tokenizer, torch, row["brief"])
            result: dict[str, Any] = {
                "brief": row["brief"],
                "reference_prompt": row["prompt"],
                "completion": completion,
                **completion_metrics(completion),
            }
            if client and api_key:
                try:
                    result["judge"] = judge_completion(
                        client,
                        api_key,
                        judge_model,
                        rubric,
                        row["brief"],
                        completion,
                    )
                    result["judge_error"] = None
                except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                    result["judge"] = {key: None for key in SCORE_KEYS}
                    result["judge_error"] = str(exc)
            results.append(result)
            print(f"Generated {index}/{len(rows)}")
    finally:
        if client:
            client.close()

    name = args.name or ("adapter" if args.adapter else "base")
    output = {
        "name": name,
        "base_model": base_model,
        "adapter": str(args.adapter) if args.adapter else None,
        "judge_model": None if args.skip_judge else judge_model,
        "count": len(results),
        "means": summarize(results, judged=not args.skip_judge),
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{name}.json"
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
