"""Load the vendored HumanEval-Python dataset and pick a subset.

Each record has: name, prompt (signature+docstring; the model input),
entry_point (function name), test (the hidden `check(candidate)` scorer).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Optional

DEFAULT_PATH = Path(__file__).parent / "data" / "humaneval-py.jsonl"


def load_dataset(path: "str | Path | None" = None) -> List[dict]:
    p = Path(path) if path else DEFAULT_PATH
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def select_subset(
    data: List[dict],
    num_problems: Optional[int] = None,
    start: int = 0,
    names: Optional[List[str]] = None,
    shuffle_seed: Optional[int] = None,
) -> List[dict]:
    """Return a slice/selection of problems.

    `names` (if given) filters by exact name. Otherwise we take `num_problems`
    starting at `start`, optionally after a seeded shuffle for a random sample.
    """
    if names:
        wanted = set(names)
        return [d for d in data if d["name"] in wanted]
    pool = list(data)
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(pool)
    pool = pool[start:]
    return pool[:num_problems] if num_problems is not None else pool
