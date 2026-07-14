"""Distill prose prompts into structured cinematic video prompts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, TextIO

import httpx


API_URL = "https://api.anthropic.com/v1/messages"
MODEL = os.environ.get("DISTILL_MODEL", "claude-sonnet-5")
MAX_TOKENS = 3000
RATE_LIMIT_SECONDS = 0.5
DATASETS = (
    (Path("data/train.jsonl"), Path("data/train_structured.jsonl")),
    (Path("data/val.jsonl"), Path("data/val_structured.jsonl")),
)
FAILURES_PATH = Path("data/distill_failures.jsonl")
BLOCK_HEADERS = ("SUBJECT", "LOCATION", "ACTION", "CAMERA", "STYLE", "CONSTRAINTS")

SYSTEM_PROMPT = """You convert cinematic prose into one standalone, copy-ready ~15-second video prompt using this exact block order. Output plain text only, without commentary or code fences.

SUBJECT — Identify every person, prop, or effect; use stable @tags for recurring assets and say tagged inputs match 100%. State goal, precise emotional beat, white balance, and MULTISHOT. Describe acting through visible micro-actions, eye-lines, breath, and pauses.
LOCATION — Treat any @location as a STYLE REFERENCE ONLY, never a fixed keyframe. Describe architecture, mood, and natural light; let the world extend and the subject travel through it. Do not reproduce a reference 1:1. Use @scheme only when spatial layout must remain fixed.
ACTION — State intent, then fill the full ~15 seconds with timecoded SHOT beats: SHOT 1 (0:00–...), SHOT 2 (...), etc. Give blocking, gestures, eye-lines, transitions, and concrete scene changes. Keep movement from frame one and no dead air. End at 0:15. Mark hard cuts or an intentional continuous take.
CAMERA — For every shot specify angle, height, physical cine-lens feel, movement, and why that movement serves the beat. Keep motion intentional and use 180-degree shutter motion blur.
STYLE — Give Dominant 60% / Secondary 30% / Accent 10% for this scene plus WB. Preserve supplied visual style while favoring 8K photoreal cinematic texture, naturalistic lighting, haze, realistic skin, 24fps smooth motion, physical gravity/contact shadows, and rule-of-thirds/golden-ratio composition; no 3D or game-render look.
CONSTRAINTS — State hard rules including 16:9, continuity/identity/prop/scale locks, legibility, camera behavior, NO eye glow, no jitter, environmental SFX only, no music or subtitles, and real gravity/inertia. Slow motion is opt-in: default NO slow-motion; if needed, name the speed-ramped beat and return to normal speed.

Keep continuity requirements inside SUBJECT and ACTION as well as hard locks. Prefer broad physical behavior over brittle exact speeds or angles. Keep concrete scale where it matters. Avoid vague emotions. Convert this cinematic prompt into the skeleton, inventing timecoded shot beats consistent with the scene; keep all concrete visual language; carry constraints through. Every response must contain SUBJECT, LOCATION, ACTION (with timecoded SHOT beats), CAMERA, STYLE (60:30:10), and CONSTRAINTS blocks."""


def validate_structured_prompt(prompt: str) -> bool:
    """Return whether prompt contains all required block headers as headers."""
    return all(
        re.search(rf"(?m)^\s*{header}\s*(?:—|:|-)\s*\S", prompt)
        for header in BLOCK_HEADERS
    )


def source_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def load_processed(path: Path) -> set[str]:
    processed: set[str] = set()
    if not path.exists():
        return processed
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                processed.add(source_hash(row["source_prompt"]))
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"Invalid resume row in {path}:{line_number}") from exc
    return processed


def request_structured_prompt(
    client: httpx.Client, api_key: str, brief: str, prompt: str
) -> tuple[str, int, int]:
    response = client.post(
        API_URL,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": MODEL,
            "max_tokens": MAX_TOKENS,
            "system": SYSTEM_PROMPT,
            "messages": [
                {
                    "role": "user",
                    "content": f"BRIEF:\n{brief}\n\nORIGINAL PROMPT:\n{prompt}",
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
    usage = payload.get("usage", {})
    return (
        "\n".join(text_parts).strip(),
        int(usage.get("input_tokens", 0)),
        int(usage.get("output_tokens", 0)),
    )


def write_jsonl_row(handle: TextIO, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def process_dataset(
    input_path: Path,
    output_path: Path,
    client: httpx.Client,
    api_key: str,
    failures: TextIO,
    stats: dict[str, int],
) -> None:
    processed_hashes = load_processed(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open(encoding="utf-8") as source, output_path.open(
        "a", encoding="utf-8"
    ) as output:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt_hash = source_hash(row["prompt"])
            stats["seen"] += 1

            if prompt_hash in processed_hashes:
                stats["skipped"] += 1
            else:
                if stats["calls"]:
                    time.sleep(RATE_LIMIT_SECONDS)
                stats["calls"] += 1
                structured: str | None = None
                try:
                    structured, input_tokens, output_tokens = request_structured_prompt(
                        client, api_key, row["brief"], row["prompt"]
                    )
                    stats["input_tokens"] += input_tokens
                    stats["output_tokens"] += output_tokens
                    if not validate_structured_prompt(structured):
                        raise ValueError("Generated prompt missing required block header(s)")

                    write_jsonl_row(
                        output,
                        {
                            "brief": row["brief"],
                            "prompt": structured,
                            "brief_source": row["brief_source"],
                            "source_prompt": row["prompt"],
                        },
                    )
                    processed_hashes.add(prompt_hash)
                    stats["passed"] += 1
                except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                    failure = {
                        "input_file": str(input_path),
                        "line": line_number,
                        "brief": row.get("brief"),
                        "brief_source": row.get("brief_source"),
                        "source_prompt": row.get("prompt"),
                        "source_hash": prompt_hash,
                        "error": str(exc),
                    }
                    if structured is not None:
                        failure["generated_prompt"] = structured
                    write_jsonl_row(failures, failure)
                    stats["failed"] += 1

            if stats["seen"] % 10 == 0:
                print(
                    f"Progress: {stats['seen']} items "
                    f"({stats['passed']} pass, {stats['failed']} fail, "
                    f"{stats['skipped']} skipped)"
                )


def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CRAG_LLM_API_KEY")
    if not api_key:
        raise SystemExit("Set ANTHROPIC_API_KEY or CRAG_LLM_API_KEY")

    stats = {
        "seen": 0,
        "calls": 0,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }
    FAILURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=120.0) as client, FAILURES_PATH.open(
        "w", encoding="utf-8"
    ) as failures:
        for input_path, output_path in DATASETS:
            process_dataset(
                input_path, output_path, client, api_key, failures, stats
            )

    estimated_cost = (
        stats["input_tokens"] * 3 / 1_000_000
        + stats["output_tokens"] * 15 / 1_000_000
    )
    print(
        f"Validation: {stats['passed']} pass, {stats['failed']} fail, "
        f"{stats['skipped']} skipped"
    )
    print(
        f"Tokens: {stats['input_tokens']} input, {stats['output_tokens']} output"
    )
    print(f"Estimated cost: ${estimated_cost:.4f}")


if __name__ == "__main__":
    main()
