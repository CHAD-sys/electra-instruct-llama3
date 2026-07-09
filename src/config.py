"""Central place for all the knobs I ended up tuning.

I kept everything here instead of hardcoding stuff in train.py because I got
tired of grepping for magic numbers every time I ran out of VRAM.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    base_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    # 4-bit is the only reason this fits on my 3090 (24GB). Turn off if you
    # have an A100 lying around (I don't).
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    # bf16 on Ampere+, fp16 otherwise. train.py picks automatically.
    max_seq_len: int = 1024


@dataclass
class LoraConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"
    # These are the ones that actually matter for Llama. I tried attention-only
    # first (q,k,v,o) and it underfit, adding the MLP projections helped a lot.
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])


@dataclass
class TrainConfig:
    output_dir: str = "outputs/llama3-electrical-lora"
    dataset_path: str = "data/electrical_instructions.jsonl"
    epochs: int = 3
    # effective batch size = per_device * grad_accum = 16
    per_device_batch_size: int = 2
    grad_accum_steps: int = 8
    lr: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    lr_scheduler: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 100
    eval_ratio: float = 0.05  # hold out 5% for a quick sanity eval
    seed: int = 42
    # paged_adamw_8bit saves a surprising amount of memory vs plain adamw
    optim: str = "paged_adamw_8bit"
    gradient_checkpointing: bool = True
