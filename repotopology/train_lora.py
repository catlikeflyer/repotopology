"""LoRA fine-tuning script for Gemma 4 E4B on RepoTopology 3-tier AST pruning trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def train_hf_peft(
    dataset_path: str,
    output_dir: str,
    base_model: str = "google/gemma-4-e4b-it",
    batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    epochs: int = 3,
    learning_rate: float = 2e-4,
    lora_r: int = 8,
    lora_alpha: int = 16,
    max_seq_length: int = 2048,
) -> None:
    """Trains a LoRA adapter using Hugging Face TRL SFTTrainer and PEFT."""
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from trl import SFTTrainer
    except ImportError as e:
        raise ImportError(
            "Missing PyTorch / Transformers / PEFT / TRL dependencies.\n"
            "Install them via: uv pip install torch transformers peft trl datasets accelerate"
        ) from e

    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load trajectories from JSONL
    data_file = Path(dataset_path)
    records = []
    for line in data_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line.strip()))

    print(f"Loaded {len(records)} trajectory records from {dataset_path}")

    # Format conversations using tokenizer chat template
    formatted_texts = []
    for r in records:
        msgs = r.get("messages", [])
        if msgs:
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            formatted_texts.append({"text": text})

    dataset = Dataset.from_list(formatted_texts)

    # Configure PEFT LoRA
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_path / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )

    print("Starting LoRA fine-tuning...")
    trainer.train()

    print(f"Saving final adapter to: {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Fine-tuning complete. Output matches competition 'adapters/main_lora/' format.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Gemma 4 E4B with LoRA on RepoTopology")
    parser.add_argument("--dataset", type=str, default="./data/sft_trajectories.jsonl", help="Path to trajectories JSONL")
    parser.add_argument("--output-dir", type=str, default="./submission/adapters/main_lora", help="Adapter destination")
    parser.add_argument("--base-model", type=str, default="google/gemma-4-e4b-it", help="Base Gemma 4 variant")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per device")
    args = parser.parse_args()

    train_hf_peft(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        base_model=args.base_model,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
