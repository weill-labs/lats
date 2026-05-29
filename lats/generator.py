"""The LLM-as-(agent, value-helper, reflector) wrappers.

Each function turns one of the prompt builders in `prompts.py` into a concrete
call on an `LLM` and parses the result into the shape the search needs:

- `internal_tests`   -> list of `assert` strings (LATS's self-generated tests)
- `simple_impl`      -> the root candidate solution
- `reflexion_impls`  -> `n` improved candidates conditioned on a branch's history
- `self_reflection`  -> a short natural-language hint about why code failed
"""

from __future__ import annotations

import ast
import random
from typing import List

from . import prompts
from .llm import LLM
from .parse import parse_code_block


def _is_valid_assert(line: str) -> bool:
    if "assert" not in line:
        return False
    try:
        ast.parse(line)
        return True
    except SyntaxError:
        return False


def internal_tests(
    llm: LLM, func_sig: str, num_tests: int, rng: "random.Random | None" = None
) -> List[str]:
    """Ask the LLM for unit tests; keep the syntactically valid asserts."""
    out = llm.chat(
        prompts.test_generation_messages(func_sig),
        n=1,
        temperature=0.0,
        max_tokens=1024,
    )[0]
    candidates = [ln.strip() for ln in out.splitlines() if _is_valid_assert(ln.strip())]
    if len(candidates) <= num_tests:
        return candidates
    return (rng or random).sample(candidates, num_tests)


def simple_impl(
    llm: LLM, func_sig: str, temperature: float = 0.0, max_tokens: int = 1024
) -> "str | None":
    """Generate the initial solution (root of the search tree)."""
    out = llm.chat(
        prompts.simple_impl_messages(func_sig),
        n=1,
        temperature=temperature,
        max_tokens=max_tokens,
    )[0]
    return parse_code_block(out)


def reflexion_impls(
    llm: LLM,
    func_sig: str,
    prev_impls: List[str],
    feedbacks: List[str],
    reflections: List[str],
    n: int,
    temperature: float = 0.8,
    max_tokens: int = 1024,
) -> List[str]:
    """Sample `n` improved solutions given a branch's accumulated history.

    Drops any completion we cannot parse into code rather than emitting None.
    """
    messages = prompts.reflexion_impl_messages(
        func_sig, prev_impls, feedbacks, reflections
    )
    outs = llm.chat(messages, n=n, temperature=temperature, max_tokens=max_tokens)
    parsed = [parse_code_block(o) for o in outs]
    return [p for p in parsed if p]


def self_reflection(llm: LLM, func: str, feedback: str, max_tokens: int = 512) -> str:
    """Explain why `func` failed `feedback` — fed into later expansions."""
    return llm.chat(
        prompts.self_reflection_messages(func, feedback),
        n=1,
        temperature=0.0,
        max_tokens=max_tokens,
    )[0].strip()
