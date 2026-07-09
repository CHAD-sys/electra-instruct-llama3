"""Load the base model + trained LoRA adapter and chat with it.

    python -m src.inference --adapter outputs/llama3-electrical-lora \
        --prompt "What does the tolerance band on a resistor tell me?"

Leave out --prompt to drop into a tiny REPL.
"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from src.config import ModelConfig
from src.dataset import SYSTEM_PROMPT


def load(adapter_dir: str, four_bit: bool = True):
    mcfg = ModelConfig()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    quant_cfg = None
    if four_bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=mcfg.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=mcfg.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=dtype,
        )

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    base = AutoModelForCausalLM.from_pretrained(
        mcfg.base_model,
        quantization_config=quant_cfg,
        torch_dtype=dtype,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate(model, tokenizer, user_msg: str, max_new_tokens: int = 512) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    out = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.3,
        top_p=0.9,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen = out[0][inputs.shape[-1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="outputs/llama3-electrical-lora")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    model, tokenizer = load(args.adapter)

    if args.prompt:
        print(generate(model, tokenizer, args.prompt, args.max_new_tokens))
        return

    print("Ask me about electrical components (Ctrl-C to quit).\n")
    while True:
        try:
            q = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        print("\nmodel >", generate(model, tokenizer, q, args.max_new_tokens), "\n")


if __name__ == "__main__":
    main()
