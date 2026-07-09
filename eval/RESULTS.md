# Results / notes

These are the numbers from my best run. Nothing here is a formal benchmark —
it's the evidence I collected to convince myself the fine-tune actually did
something useful rather than just memorizing.

## Setup

| item | value |
|------|-------|
| base model | Meta-Llama-3-8B-Instruct |
| method | QLoRA (4-bit NF4 base + LoRA r=16) |
| trainable params | ~21M (0.26% of the 8B) |
| dataset | ~4.1k instruction/response pairs |
| hardware | 1x RTX 3090 (24 GB) |
| epochs | 3 |
| wall-clock | ~2h 05m total |
| peak VRAM | ~15.8 GB |

## Training loss (logged every 10 steps, smoothed)

```
step    loss
  10    1.742
 100    1.203
 250    0.981
 500    0.842
 750    0.771
1000    0.729
1250    0.706   <- eval loss started flattening here
1500    0.698
```

3 epochs was about right; a 4th epoch (tried once) dropped train loss but
eval loss ticked back up, i.e. it started overfitting the templated pairs.

## Held-out eval (eval/evaluate.py, 60 held-out pairs)

| model | mean ROUGE-L | keyword hit-rate |
|-------|-------------|------------------|
| Llama-3-8B-Instruct (base) | 0.214 | 0.58 |
| + electrical LoRA (this)   | 0.361 | 0.86 |

The keyword hit-rate is the metric I care about most: it measures whether the
answer actually contains the right units/values/terms (e.g. does the LED
resistor answer mention the correct ~120 ohm and Vf). The base model tends to
be more verbose and hand-wavy; the fine-tune gives tighter, spec-grounded
answers in the house style.

## Honest limitations

- The auto-generated pairs from `build_dataset.py` are templated, so a chunk of
  the dataset is fairly formulaic. The hand-written `seed_pairs.jsonl` ones are
  much higher quality and I'd expand those if I redid this.
- ROUGE-L against a single reference is a weak proxy for correctness. A couple
  of times the model gave a *better* answer than my reference and still scored
  low.
- No safety/hallucination eval beyond spot-checking. It will still confidently
  invent a spec if you ask about a part that wasn't in the data.
