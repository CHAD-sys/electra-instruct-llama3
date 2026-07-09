"""Pure, dependency-free metric functions.

Pulled out of evaluate.py so they can be unit-tested without importing torch /
transformers (which evaluate.py needs for the model).
"""
from typing import Iterable


def rouge_l(pred: str, ref: str) -> float:
    """Longest-common-subsequence F1 at the word level."""
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


def keyword_hit(pred: str, keywords: Iterable[str]) -> float:
    """Fraction of expected keywords present in the prediction (0..1)."""
    keywords = list(keywords)
    if not keywords:
        return 1.0
    pred_l = pred.lower()
    hits = sum(1 for k in keywords if k.lower() in pred_l)
    return hits / len(keywords)
