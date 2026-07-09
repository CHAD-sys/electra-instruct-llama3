"""Merge the LoRA adapter into the base weights and save a standalone model.

Useful if you want to export to GGUF/llama.cpp later or serve without PEFT.
Note: merging needs the base model in fp16/bf16 (not 4-bit), so this wants
more RAM/VRAM than training did. I ran it on CPU with 32GB RAM when my GPU
was busy.

    python -m src.merge_adapter --adapter outputs/llama3-electrical-lora \
        --out outputs/llama3-electrical-merged
"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.config import ModelConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="outputs/llama3-electrical-lora")
    ap.add_argument("--out", default="outputs/llama3-electrical-merged")
    args = ap.parse_args()

    mcfg = ModelConfig()
    print("[merge] loading base model in fp16 (this is memory heavy)...")
    base = AutoModelForCausalLM.from_pretrained(
        mcfg.base_model, torch_dtype=torch.float16, device_map="cpu"
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    print("[merge] merging adapter weights...")
    model = model.merge_and_unload()

    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(args.out)
    print(f"[merge] merged model saved to {args.out}")


if __name__ == "__main__":
    main()
