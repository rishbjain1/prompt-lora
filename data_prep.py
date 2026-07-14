"""Extract privacy-filtered prompt tuning pairs from the local prompt corpus."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


SOURCE_FILES = (
    Path("/Users/rishabhjain/AI content/Prompts_Vault.md"),
    Path("/Users/rishabhjain/AI content/Cinematic_Prompt_Library.md"),
    Path("/Users/rishabhjain/AI content/Shot_Recipes.md"),
    Path("/Users/rishabhjain/AI content/Storyboards_Scenes.md"),
)

PRIVATE_TOPICS = {
    "career": re.compile(r"\bcareer\b", re.IGNORECASE),
    "visa": re.compile(r"\bvisa\b", re.IGNORECASE),
    "o-1b": re.compile(r"\bo[\s-]?1b\b", re.IGNORECASE),
    "immigration": re.compile(r"\bimmigration\b", re.IGNORECASE),
    "teleparty": re.compile(r"\bteleparty\b", re.IGNORECASE),
}

# Curated from permitted corpus plus user-provided private-name examples. Generic
# character labels (woman, model, courier, hero) remain valid training content.
PERSON_NAMES = (
    "Rishabh",
    "Shuchi",
    "Rahul",
    "Shaurya",
    "Youssuf",
    "Helmut Newton",
    "Juergen Teller",
    "Solve Sundsbo",
    "Richard Avedon",
    "Donatella",
    "Mert & Marcus",
    "Mario Testino",
    "Ottessa Moshfegh",
    "Wes Anderson",
    "Tony Scott",
    "Joseph Kosinski",
    "Caravaggio",
    "Henri Cartier-Bresson",
    "Gordon Parks",
    "Steve McCurry",
    "Tolkien",
)
PERSON_PATTERNS = {
    name.lower(): re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE)
    for name in PERSON_NAMES
}


def apply_privacy_filter(text: str, hits: Counter[str]) -> str | None:
    """Reject private topics; remove lines containing known real-person names."""
    for label, pattern in PRIVATE_TOPICS.items():
        if pattern.search(text):
            hits[f"topic:{label}"] += 1
            return None

    kept: list[str] = []
    for line in text.splitlines():
        matched = [label for label, pattern in PERSON_PATTERNS.items() if pattern.search(line)]
        if matched:
            for label in matched:
                hits[f"person:{label}"] += 1
            continue
        kept.append(line.rstrip())

    cleaned = "\n".join(kept).strip()
    return cleaned or None


def derived_brief(title: str, prompt: str) -> str:
    """Build a short deterministic brief from title and first content sentence."""
    plain = re.sub(r"\s+", " ", prompt).strip().strip('"')
    first = re.split(r"(?<=[.!?])\s+", plain, maxsplit=1)[0]
    words = first.split()
    if len(words) > 36:
        first = " ".join(words[:36]).rstrip(",;:") + "."
    title = re.sub(r"\s*\([^)]*\)\s*$", "", title).strip()
    return f"{title}: {first}" if title else first


def build_pair(
    title: str,
    prompt: str,
    explicit_brief: str | None,
    hits: Counter[str],
) -> dict[str, str] | None:
    combined = f"{explicit_brief or ''}\n{prompt}"
    for label, pattern in PRIVATE_TOPICS.items():
        if pattern.search(combined):
            hits[f"topic:{label}"] += 1
            return None

    clean_prompt = apply_privacy_filter(prompt, hits)
    if not clean_prompt:
        return None

    if explicit_brief:
        clean_brief = apply_privacy_filter(explicit_brief.strip(), hits)
        if clean_brief:
            source = "explicit"
        else:
            clean_brief = apply_privacy_filter(derived_brief(title, clean_prompt), hits)
            source = "derived"
    else:
        clean_brief = apply_privacy_filter(derived_brief(title, clean_prompt), hits)
        source = "derived"
    if not clean_brief:
        return None

    return {"brief": clean_brief, "prompt": clean_prompt, "brief_source": source}


def markdown_sections(text: str) -> Iterable[tuple[str, str, str]]:
    """Yield (level-two section, level-three title, body)."""
    section = ""
    title: str | None = None
    body: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if title is not None:
                yield section, title, "\n".join(body)
            section = line[3:].strip()
            title = None
            body = []
        elif line.startswith("### "):
            if title is not None:
                yield section, title, "\n".join(body)
            title = line[4:].strip()
            body = []
        elif title is not None:
            body.append(line)
    if title is not None:
        yield section, title, "\n".join(body)


def extract_vault(path: Path) -> tuple[list[dict[str, str]], Counter[str]]:
    pairs: list[dict[str, str]] = []
    hits: Counter[str] = Counter()
    text = path.read_text(encoding="utf-8")

    for section, title, body in markdown_sections(text):
        if section == "ChatGPT meta-prompts":
            continue

        use_case = re.search(r"\*\*Use case:\*\*\s*([^\n|]+)", body)
        explicit = use_case.group(1).strip() if use_case else None

        quote_lines = []
        for line in body.splitlines():
            if line.startswith(">"):
                quote_lines.append(line[1:].lstrip())
        prompt = "\n".join(quote_lines).strip()
        if prompt:
            pair = build_pair(title, prompt, explicit, hits)
            if pair:
                pairs.append(pair)

        for bullet in re.findall(r'^-\s+"(.+?)"\s*$', body, re.MULTILINE):
            pair = build_pair(title, bullet, None, hits)
            if pair:
                pairs.append(pair)

    return pairs, hits


def extract_shot_recipes(path: Path) -> tuple[list[dict[str, str]], Counter[str]]:
    pairs: list[dict[str, str]] = []
    hits: Counter[str] = Counter()
    text = path.read_text(encoding="utf-8")

    for match in re.finditer(
        r"^##\s+(.+?)\n(?P<body>.*?)(?=^##\s+|\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    ):
        title = match.group(1).strip()
        body = match.group("body")
        what = re.search(r"^\*\*What:\*\*\s*(.+)$", body, re.MULTILINE)
        brief = what.group(1).strip() if what else None
        for prompt in re.findall(r"```(?:[^\n]*)\n(.*?)```", body, re.DOTALL):
            pair = build_pair(title, prompt.strip(), brief, hits)
            if pair:
                pairs.append(pair)

    return pairs, hits


def extract_source(path: Path) -> tuple[list[dict[str, str]], Counter[str]]:
    if path.name == "Prompts_Vault.md":
        return extract_vault(path)
    if path.name == "Shot_Recipes.md":
        return extract_shot_recipes(path)
    # Remaining files are vocabulary/empty storyboard references, not examples.
    return [], Counter()


def split_pairs(
    pairs: list[dict[str, str]], seed: int = 42
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    val_size = max(1, int(len(shuffled) * 0.1 + 0.5)) if shuffled else 0
    return shuffled[val_size:], shuffled[:val_size]


def write_jsonl(path: Path, rows: Iterable[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    all_pairs: list[dict[str, str]] = []
    all_hits: Counter[str] = Counter()
    counts: dict[str, int] = {}
    for source in SOURCE_FILES:
        if not source.is_file():
            raise FileNotFoundError(f"Missing source corpus: {source}")
        pairs, hits = extract_source(source)
        counts[source.name] = len(pairs)
        all_pairs.extend(pairs)
        all_hits.update(hits)

    train, val = split_pairs(all_pairs, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "val.jsonl", val)

    print("Extraction summary")
    for source, count in counts.items():
        print(f"  {source}: {count}")
    print(f"  total: {len(all_pairs)}")
    print(f"  train: {len(train)}")
    print(f"  val: {len(val)}")
    print("Privacy filter hits")
    if all_hits:
        for label, count in sorted(all_hits.items()):
            print(f"  {label}: {count}")
    else:
        print("  none")


if __name__ == "__main__":
    main()
