# electra-instruct-llama3

Fine-tuning **Llama 3 8B Instruct** to be a domain assistant for **electrical &
electronic components**, using **parameter-efficient fine-tuning (QLoRA)** so the
whole thing trains on a single consumer GPU instead of a datacenter.

I built this to learn how instruction fine-tuning actually works end-to-end —
from turning messy technical docs into training data, to training a LoRA adapter,
to evaluating whether it made the model any better. It did (see
[`eval/RESULTS.md`](eval/RESULTS.md)).

> **Why not just prompt the base model?** You can, and for a one-off question
> prompting is fine. The point of this project was to see how much a small,
> cheap LoRA adapter can push a general model toward a specific domain's
> vocabulary and answer style without full retraining. Turns out: quite a bit.

---

## What it does

Ask it component questions and it answers in a consistent, spec-grounded style:

```
you > How do I size the series resistor for a red LED on 5V at 10 mA?

model > Use R = (Vsupply - Vf) / I. A red LED has Vf ≈ 2.0 V, so
        R = (5 - 2.0) / 0.010 = 300 Ω. The nearest standard value is 300 Ω
        (or 330 Ω for margin). Power = I²R = 30 mW, so a 1/4 W resistor is fine.
```

```
you > Can I drive an IRF540N straight from a 3.3V GPIO pin?

model > Not reliably. The IRF540N is a standard-level MOSFET (VGS(th) 2–4 V)
        and needs about 10 V on the gate to fully turn on. At 3.3 V it stays
        partially on with high RDS(on) and overheats. Use a logic-level MOSFET
        or a gate driver.
```

## Why QLoRA (the whole point)

Full fine-tuning of an 8B model needs ~60–80 GB of VRAM — not happening on my
machine. QLoRA gets it down to a single 24 GB card by:

1. Loading the base model in **4-bit** (NF4 quantization) so the frozen weights
   take ~5 GB instead of ~16 GB.
2. Training only a small set of **LoRA adapter** matrices (rank 16) injected into
   the attention + MLP projections — about **21M trainable params, 0.26%** of the
   model.
3. Using a paged 8-bit optimizer + gradient checkpointing to fit the rest.

Result: trains in ~2 hours on an RTX 3090, peak VRAM ~15.8 GB. The output is a
~80 MB adapter you load on top of the stock Llama 3, not a new 16 GB model.

## How the data is made

I didn't want to hand-write thousands of examples, so the pipeline is:

- `data/seed_pairs.jsonl` — hand-written, high-quality Q/A pairs (the good stuff).
- `data/raw/*.txt` — component docs in a simple `name / key: value / notes`
  format (a few sample ones are committed so the pipeline runs out of the box).
- `data/build_dataset.py` — parses each doc, generates templated
  instruction/response pairs from the specs and notes, mixes in the seed pairs,
  de-duplicates, shuffles, and writes `data/electrical_instructions.jsonl`.

```bash
python data/build_dataset.py            # -> data/electrical_instructions.jsonl
```

Everything is deterministic (fixed seed) so the dataset is reproducible.

## Project layout

```
electra-instruct-llama3/
├── configs/lora.yaml         # readable copy of the hyperparameters
├── data/
│   ├── build_dataset.py      # docs -> instruction/response pairs
│   ├── seed_pairs.jsonl      # hand-written examples
│   ├── eval_held_out.jsonl   # held-out eval set
│   └── raw/*.txt             # sample component docs
├── src/
│   ├── config.py             # all hyperparameters live here
│   ├── dataset.py            # chat-template formatting
│   ├── train.py              # QLoRA training loop (TRL SFTTrainer)
│   ├── inference.py          # load base + adapter, chat / one-shot
│   └── merge_adapter.py      # bake adapter into base for export
├── eval/
│   ├── evaluate.py           # ROUGE-L + keyword hit-rate
│   └── RESULTS.md            # numbers + honest limitations
└── scripts/                  # train.sh / chat.sh convenience wrappers
```

## Setup

You need a CUDA GPU with ~16 GB VRAM and access to the gated Llama 3 weights on
Hugging Face.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# accept the Llama 3 license on HF, then:
huggingface-cli login
```

## Train

```bash
bash scripts/train.sh
# or directly:
python -m src.train --epochs 3
```

The adapter is saved to `outputs/llama3-electrical-lora/`.

## Chat with it

```bash
bash scripts/chat.sh
# or a one-shot question:
python -m src.inference --prompt "What does RDS(on) mean and why does it matter?"
```

## Evaluate

```bash
python eval/evaluate.py --adapter outputs/llama3-electrical-lora \
    --test data/eval_held_out.jsonl
```

| model | mean ROUGE-L | keyword hit-rate |
|-------|-------------|------------------|
| Llama-3-8B-Instruct (base) | 0.214 | 0.58 |
| + electrical LoRA (this)   | **0.361** | **0.86** |

More detail and the loss curve are in [`eval/RESULTS.md`](eval/RESULTS.md).

## Things I learned / would do next

- Attention-only LoRA (q,k,v,o) underfit; adding the MLP projections
  (gate/up/down) made a clear difference.
- The templated data works but is formulaic — the next step is generating more
  diverse instructions (paraphrasing, multi-turn, "explain the trade-off"
  questions) rather than one-fact lookups.
- Would like to export to GGUF and run it in llama.cpp on CPU for a fully
  offline component helper.

## Acknowledgements / references

- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021)
- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* (2023)
- Hugging Face `transformers`, `peft`, and `trl` — the training loop is built on
  TRL's `SFTTrainer`.
- Meta's Llama 3 8B Instruct as the base model.

## License

MIT — see [LICENSE](LICENSE). Note this only covers *my* code; the Llama 3 base
model is governed by Meta's Llama 3 Community License.
