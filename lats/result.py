"""Shared result type for every search strategy.

Keeping this in its own module lets `mcts.py` (LATS) and `strategies.py`
(simple / reflexion / dfs) return the same shape without importing each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class ProblemResult:
    name: str
    solved: bool
    final_code: str
    strategy: str = "lats"
    solved_on_first_try: bool = False
    iterations_used: int = 0
    num_candidates: int = 1
    internal_tests: List[str] = field(default_factory=list)
