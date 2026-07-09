"""Quick held-out evaluation of the fine-tuned adapter.

I'm not claiming this is a rigorous benchmark - it's a lightweight check I used
to see whether the LoRA actually helped vs the stock model. It computes:
  * token-level ROUGE-L against reference answers
  * a keyword-hit rate (does the answer mention the expected units/terms?)

    python eval/evaluate.py --adapter outputs/llama3-electrical-lora \
        --test data/eval_held_out.jsonl
"""
import argparse
import json

from src.inference import load, generate


def rouge_l(pred: str, ref: str) -> float:
    """Longest-common-subsequence F1 at the word level. No external deps."""
    a, b = pred.lower().split(), ref.lower().split()
    if not a or not b:
        return 0.0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[len(a)][len(b)]
    prec, rec = lcs / len(a), lcs / len(b)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def keyword_hit(pred: str, keywords) -> float:
    if not keywords:
        return 1.0
    pred_l = pred.lower()
    hits = sum(1 for k in keywords if k.lower() in pred_l)
    return hits / len(keywords)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="outputs/llama3-electrical-lora")
    ap.add_argument("--test", default="data/eval_held_out.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    model, tokenizer = load(args.adapter)

    rows = []
    with open(args.test, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]

    rouge_scores, kw_scores = [], []
    for ex in rows:
        user = ex["instruction"]
        if ex.get("input"):
            user += "\n\nContext:\n" + ex["input"]
        pred = generate(model, tokenizer, user, max_new_tokens=256)
        r = rouge_l(pred, ex["output"])
        k = keyword_hit(pred, ex.get("keywords", []))
        rouge_scores.append(r)
        kw_scores.append(k)
        print(f"- ROUGE-L={r:.2f} kw={k:.2f} | {ex['instruction'][:60]}")

    n = len(rows)
    print("\n=== summary ===")
    print(f"examples     : {n}")
    print(f"mean ROUGE-L : {sum(rouge_scores)/n:.3f}")
    print(f"mean keyword : {sum(kw_scores)/n:.3f}")


if __name__ == "__main__":
    main()
