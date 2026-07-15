# prompt-lora

QLoRA pipeline teaching Mistral 7B Instruct to turn short creative briefs into structured cinematic video prompts. Target blocks: `SUBJECT`, `LOCATION`, timecoded `ACTION`/`SHOT`s, `CAMERA`, `STYLE` 60:30:10, and `CONSTRAINTS`.

## Install

Data preparation, distillation, and evaluation need Python 3.10+:

```bash
pip install -e .
```

Local training additionally needs:

```bash
pip install -e '.[train]'
```

Colab installs training packages in the notebook.

## 1. Prepare source pairs

```bash
python data_prep.py
```

This reads the fixed, read-only corpus inputs and writes seeded 90/10 splits:

- `data/train.jsonl`
- `data/val.jsonl`

Rows contain `brief`, source `prompt`, and `brief_source` (`explicit` or `derived`). Extraction filters known private topics and curated real-person names. Review output before sharing or training; this is conservative string filtering, not general named-entity recognition.

## 2. Distill structured targets

```bash
export ANTHROPIC_API_KEY=...
python distill.py
```

`CRAG_LLM_API_KEY` is accepted instead. Distillation writes:

- `data/train_structured.jsonl`
- `data/val_structured.jsonl`

Structured row schema:

```json
{"brief": "...", "prompt": "...", "brief_source": "explicit", "source_prompt": "..."}
```

This is teacher–student training: Claude rewrites source prompts into structured targets, then open-weight model learns those Claude-distilled outputs. Results therefore measure imitation of teacher format and judgment, not independent ground truth. Inspect distilled samples and failures before training.

## 3. Train QLoRA on Colab

Open [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb), select a T4 runtime, and run all cells. Upload `train_qlora.py`, `eval_prompts.py`, `distill.py`, and both structured datasets when prompted. Notebook generates offline base completions, trains, generates offline adapter completions, then downloads `prompt-lora-run.zip` containing `out/adapter/` and `eval/out/*.json`. Training logic remains in `train_qlora.py`.

Direct invocation:

```bash
python train_qlora.py
```

Default base model is `mistralai/Mistral-7B-Instruct-v0.3`; override with `BASE_MODEL`. CUDA uses 4-bit NF4 QLoRA. Apple Silicon MPS and systems without bitsandbytes use unquantized weights with a warning. Adapter and tokenizer save to `out/adapter/`.

Defaults: 3 epochs, learning rate `2e-4`, cosine schedule, LoRA `r=16`, alpha `32`, dropout `0.05`, batch size 1, gradient accumulation 8, gradient checkpointing, and attention `q/k/v/o` projections.

## 4. Evaluate before and after

Generate and judge base completions:

```bash
export ANTHROPIC_API_KEY=...
python eval_prompts.py --name base
```

Generate and judge adapter completions against identical validation briefs:

```bash
python eval_prompts.py --adapter out/adapter --name adapter
```

Set `JUDGE_MODEL` to override default `claude-haiku-4-5-20251001`. `CRAG_LLM_API_KEY` is accepted instead of `ANTHROPIC_API_KEY`. Each run writes `eval/out/<name>.json` with full generated completions, per-example judge scores, judge errors, block-header validity, and means. CUDA generation uses 4-bit NF4 weights when bitsandbytes is available.

Offline-only evaluation still generates model completions but skips Anthropic calls:

```bash
python eval_prompts.py --adapter out/adapter --name adapter-offline --skip-judge
```

Block-header validity is free and uses `distill.validate_structured_prompt`. Validation data comes from the same source corpus, so scores measure held-out format/style fit rather than broad generalization.

## Results

| Model | Skeleton | Shot grammar | Constraints | Specificity | Total | Valid headers |
|---|---:|---:|---:|---:|---:|---:|
| Base | — | — | — | — | — | — |
| Base + LoRA | — | — | — | — | — | — |

## Tests

```bash
pytest -q
```

Tests are offline: no model loading, GPU, training, or network calls.
