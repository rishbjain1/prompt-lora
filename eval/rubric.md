# Cinematic prompt judge rubric

Score only generated prompt. Compare it with input brief. Award 0–10 points; half-points allowed.

## 1. Skeleton compliance — 0–3

- **3:** Exact readable blocks: `SUBJECT`, `LOCATION`, `ACTION`, `CAMERA`, `STYLE`, `CONSTRAINTS`. `ACTION` contains multiple `SHOT` entries. `STYLE` gives 60:30:10 proportions.
- **2:** All core blocks present, but one block weak/misordered or style proportions incomplete.
- **1:** Three to five blocks present; structure hard to use.
- **0:** Unstructured prose or fewer than three required blocks.

## 2. Shot grammar — 0–2.5

- **2.5:** Shots have valid, non-overlapping timestamps that cover intended clip; blocking, action, eye-line, and transitions are clear. Camera moves are physically plausible and motivated.
- **1.5:** Mostly usable timing and camera language; minor gaps, overlaps, or generic moves.
- **0.5:** Shot labels exist but timing or movement is ambiguous/contradictory.
- **0:** No timecoded shots.

Valid moves include locked-off, handheld, pan, tilt, dolly/push/pull, tracking, crane/jib, arc/orbit, FPV, and macro moves when physically motivated. Penalize impossible simultaneous moves or unexplained movement stacks.

## 3. Constraint adherence — 0–2.5

- **2.5:** Carries every brief constraint through relevant blocks: aspect ratio, palette/60:30:10 colors, audio/music/dialogue, duration, speed, continuity, legibility, and exclusions. No contradictions.
- **1.5:** Preserves most constraints; one material omission or mild contradiction.
- **0.5:** Several constraints missing or only repeated without operational detail.
- **0:** Violates central brief constraint or invents conflicting rules.

If brief omits a category, judge sensible defaults rather than requiring invented constraints.

## 4. Specificity — 0–2

- **2:** Concrete subjects, gestures, spatial relations, light direction, lens/framing, textures, atmosphere, and visible cause/effect. Details support brief.
- **1:** Mix of concrete and vague language; usable but interchangeable.
- **0:** Mostly abstract adjectives, style-name stacking, or unsupported filler.

## Final judgment

Return JSON only:

```json
{
  "skeleton_compliance": 0,
  "shot_grammar": 0,
  "constraint_adherence": 0,
  "specificity": 0,
  "total": 0,
  "reason": "One concise evidence-based sentence."
}
```

`total` must equal component sum and remain within 0–10. Judge baseline and fine-tuned outputs blind, with randomized labels and identical temperature/model settings.
