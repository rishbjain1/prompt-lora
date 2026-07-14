"""Fine-tune a causal language model on distilled brief/prompt pairs."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any


DEFAULT_BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
TRAIN_PATH = Path("data/train_structured.jsonl")
OUTPUT_DIR = Path("out/adapter")
USER_INSTRUCTION = (
    "Convert this creative brief into one standalone, copy-ready structured "
    "cinematic video prompt using SUBJECT, LOCATION, ACTION, CAMERA, STYLE, "
    "and CONSTRAINTS blocks."
)


def format_chat(tokenizer: Any, row: dict[str, str]) -> str:
    """Render one training pair with model-native chat formatting."""
    messages = [
        {
            "role": "user",
            "content": f"{USER_INSTRUCTION}\n\n{row['brief']}",
        },
        {"role": "assistant", "content": row["prompt"]},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def load_model_and_tokenizer(base_model: str):
    """Load base model with CUDA QLoRA or an unquantized fallback."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    use_cuda = torch.cuda.is_available()
    use_mps = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    model_kwargs: dict[str, Any] = {}
    quantized = False

    if use_cuda:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            warnings.warn(
                "bitsandbytes unavailable; loading unquantized model. GPU memory use will be much higher.",
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
            quantized = True
    elif use_mps:
        warnings.warn(
            "CUDA unavailable; loading unquantized model on Apple Silicon MPS.",
            stacklevel=2,
        )
        model_kwargs["torch_dtype"] = torch.float16
    else:
        warnings.warn(
            "CUDA and MPS unavailable; loading unquantized model on CPU.",
            stacklevel=2,
        )
        model_kwargs["torch_dtype"] = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    if use_mps:
        model.to("mps")
    model.config.use_cache = False
    if quantized:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )

    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    return model, tokenizer, use_bf16, use_cuda and not use_bf16, quantized


def main() -> None:
    if not TRAIN_PATH.exists():
        raise SystemExit(f"Missing training data: {TRAIN_PATH}")

    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    base_model = os.environ.get("BASE_MODEL", DEFAULT_BASE_MODEL)
    model, tokenizer, use_bf16, use_fp16, quantized = load_model_and_tokenizer(
        base_model
    )
    dataset = load_dataset("json", data_files=str(TRAIN_PATH), split="train")
    dataset = dataset.map(
        lambda row: {"text": format_chat(tokenizer, row)},
        remove_columns=dataset.column_names,
    )

    args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=3,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
        seed=42,
        max_length=2048,
        dataset_text_field="text",
        optim="paged_adamw_8bit" if quantized else "adamw_torch",
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))


if __name__ == "__main__":
    main()
