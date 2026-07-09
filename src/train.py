"""QLoRA fine-tuning entry point.

Run:  python -m src.train
or:   bash scripts/train.sh

This trains a LoRA adapter on top of a 4-bit quantized Llama 3 8B. On my
RTX 3090 (24GB) one epoch over ~4k examples takes roughly 40 min.
"""
import argparse
import os

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from peft import LoraConfig as PeftLoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

from src.config import ModelConfig, LoraConfig, TrainConfig
from src.dataset import load_dataset


def pick_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_base_model(mcfg: ModelConfig, dtype: torch.dtype):
    quant_cfg = None
    if mcfg.load_in_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=mcfg.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=mcfg.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=dtype,
        )

    tokenizer = AutoTokenizer.from_pretrained(mcfg.base_model)
    # Llama 3 has no pad token by default; reuse eos so batching works.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        mcfg.base_model,
        quantization_config=quant_cfg,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, help="override dataset path")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    mcfg, lcfg, tcfg = ModelConfig(), LoraConfig(), TrainConfig()
    if args.dataset:
        tcfg.dataset_path = args.dataset
    if args.epochs:
        tcfg.epochs = args.epochs
    if args.output_dir:
        tcfg.output_dir = args.output_dir

    set_seed(tcfg.seed)
    dtype = pick_dtype()
    print(f"[train] compute dtype = {dtype}")

    model, tokenizer = load_base_model(mcfg, dtype)
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=tcfg.gradient_checkpointing
    )

    peft_cfg = PeftLoraConfig(
        r=lcfg.r,
        lora_alpha=lcfg.alpha,
        lora_dropout=lcfg.dropout,
        bias=lcfg.bias,
        task_type="CAUSAL_LM",
        target_modules=lcfg.target_modules,
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    full = load_dataset(tcfg.dataset_path, tokenizer)
    split = full.train_test_split(test_size=tcfg.eval_ratio, seed=tcfg.seed)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"[train] {len(train_ds)} train / {len(eval_ds)} eval examples")

    sft_cfg = SFTConfig(
        output_dir=tcfg.output_dir,
        num_train_epochs=tcfg.epochs,
        per_device_train_batch_size=tcfg.per_device_batch_size,
        per_device_eval_batch_size=tcfg.per_device_batch_size,
        gradient_accumulation_steps=tcfg.grad_accum_steps,
        gradient_checkpointing=tcfg.gradient_checkpointing,
        learning_rate=tcfg.lr,
        warmup_ratio=tcfg.warmup_ratio,
        weight_decay=tcfg.weight_decay,
        lr_scheduler_type=tcfg.lr_scheduler,
        logging_steps=tcfg.logging_steps,
        save_steps=tcfg.save_steps,
        eval_strategy="steps",
        eval_steps=tcfg.save_steps,
        optim=tcfg.optim,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        max_seq_length=mcfg.max_seq_len,
        dataset_text_field="text",
        packing=False,
        report_to="none",
        seed=tcfg.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    trainer.train()

    os.makedirs(tcfg.output_dir, exist_ok=True)
    trainer.save_model(tcfg.output_dir)
    tokenizer.save_pretrained(tcfg.output_dir)
    print(f"[train] adapter saved to {tcfg.output_dir}")


if __name__ == "__main__":
    main()
