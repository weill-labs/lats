"""Answer scoring for HotPotQA — the standard SQuAD/HotPotQA EM and F1 metrics.

`normalize_answer` lowercases, strips punctuation and articles, and collapses
whitespace, so "The White House." and "white house" compare equal. The terminal
reward for the search is exact match (1.0/0.0); F1 is reported alongside.
"""

from __future__ import annotations

import collections
import re
import string

_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(prediction: str, gold: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(gold)


def f1_score(prediction: str, gold: str) -> float:
    pred_toks = normalize_answer(prediction).split()
    gold_toks = normalize_answer(gold).split()
    if not pred_toks or not gold_toks:
        # 1.0 only if both are empty after normalization, else 0.0.
        return float(pred_toks == gold_toks)
    common = collections.Counter(pred_toks) & collections.Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)
