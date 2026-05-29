"""Load HotPotQA questions.

Accepts the official HotPotQA JSON (a top-level array of objects with `_id`,
`question`, `answer`, …) *or* a `.jsonl` file with one object per line. We
normalize to `{id, question, answer}`. A tiny vendored sample ships for tests and
plumbing smoke runs; point `--dataset` at the official dev file for a real eval.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

DEFAULT_PATH = Path(__file__).parent / "data" / "hotpotqa_sample.jsonl"


def _normalize(d: dict) -> dict:
    return {
        "id": str(d.get("id") or d.get("_id") or ""),
        "question": d["question"],
        "answer": d["answer"],
    }


def load_hotpotqa(path: "str | Path | None" = None) -> List[dict]:
    p = Path(path) if path else DEFAULT_PATH
    text = p.read_text()
    stripped = text.lstrip()
    if stripped.startswith("["):  # official format: one big JSON array
        return [_normalize(d) for d in json.loads(text)]
    return [_normalize(json.loads(line)) for line in text.splitlines() if line.strip()]


def select_subset(
    data: List[dict],
    num_problems: Optional[int] = None,
    start: int = 0,
    ids: Optional[List[str]] = None,
) -> List[dict]:
    if ids:
        want = set(ids)
        return [d for d in data if d["id"] in want]
    pool = data[start:]
    return pool[:num_problems] if num_problems is not None else pool
