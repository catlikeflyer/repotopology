"""Turnkey MLX LoRA fine-tuning script for Gemma 4 E4B on RepoTopology trajectories."""

from __future__ import annotations

import argparse
import json
import os
import types
from pathlib import Path

import mlx.core as mx
import mlx_lm
from mlx_lm.lora import CONFIG_DEFAULTS, load_dataset, train_model
from mlx_lm.utils import _download, load_model, load_tokenizer


def run_mlx_lora(
    model_id_or_path: str = "google/gemma-4-E4B-it",
    data_dir: str = "./data/mlx",
    adapter_path: str = "./submission/adapters/main_lora",
    num_layers: int = 16,
    batch_size: int = 1,
    iters: int = 100,
    learning_rate: float = 1e-4,
    max_seq_length: int = 1024,
) -> None:
    """Fine-tunes Gemma 4 E4B using MLX on Apple Silicon with strict=False compatibility."""
    out_dir = Path(adapter_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer for: {model_id_or_path}")
    tokenizer = load_tokenizer(model_id_or_path)

    model_path = _download(model_id_or_path)
    print(f"Loading pretrained model from {model_path} with strict=False...")
    model, config = load_model(model_path, strict=False)

    args_dict = dict(CONFIG_DEFAULTS)
    args_dict.update({
        "model": model_id_or_path,
        "data": data_dir,
        "adapter_path": str(out_dir),
        "train": True,
        "test": False,
        "batch_size": batch_size,
        "num_layers": num_layers,
        "iters": iters,
        "learning_rate": learning_rate,
        "max_seq_length": max_seq_length,
        "steps_per_report": 10,
        "steps_per_eval": 25,
        "save_every": 50,
        "val_batches": 2,
        "seed": 42,
        "mask_prompt": False,
        "grad_checkpoint": True,
        "fine_tune_type": "lora",
        "optimizer": "adamw",
        "lora_parameters": {
            "rank": 8,
            "alpha": 16,
            "dropout": 0.0,
            "scale": 2.0,
        },
    })
    args = types.SimpleNamespace(**args_dict)

    print("Loading trajectory dataset...")
    train_set, valid_set, _ = load_dataset(args, tokenizer)
    print(f"Loaded {len(train_set)} train samples and {len(valid_set)} valid samples.")

    print(f"Starting LoRA fine-tuning for {iters} iterations on Apple Silicon GPU...")
    train_model(args, model, train_set, valid_set)

    # Ensure competition adapter filename format
    mlx_adapter_file = out_dir / "adapters.safetensors"
    peft_adapter_file = out_dir / "adapter_model.safetensors"
    if mlx_adapter_file.exists() and not peft_adapter_file.exists():
        import shutil
        shutil.copy2(mlx_adapter_file, peft_adapter_file)

    # Save adapter config matching competition format
    peft_config = {
        "base_model_name_or_path": model_id_or_path,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "target_modules": ["q_proj", "o_proj"],
    }
    (out_dir / "adapter_config.json").write_text(json.dumps(peft_config, indent=2))
    print(f"Fine-tuning complete. Adapter saved to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MLX LoRA Fine-Tuning for Gemma 4 E4B")
    parser.add_argument("--model", type=str, default="google/gemma-4-E4B-it", help="Model ID or local path")
    parser.add_argument("--data", type=str, default="./data/mlx", help="Directory with train.jsonl & valid.jsonl")
    parser.add_argument("--adapter-path", type=str, default="./submission/adapters/main_lora", help="Output adapter dir")
    parser.add_argument("--iters", type=int, default=100, help="Training iterations (default: 100)")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size (default: 2)")
    parser.add_argument("--num-layers", type=int, default=16, help="Number of layers to train (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--max-seq-length", type=int, default=1024, help="Max sequence length (default: 1024)")
    args = parser.parse_args()

    run_mlx_lora(
        model_id_or_path=args.model,
        data_dir=args.data,
        adapter_path=args.adapter_path,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        iters=args.iters,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_length,
    )


if __name__ == "__main__":
    main()
