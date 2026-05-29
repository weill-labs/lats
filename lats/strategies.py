"""Baseline search strategies for head-to-head comparison with LATS.

All three reuse the same `generator`, `executor`, and prompts as `mcts.py`, so a
comparison isolates the *search* differences:

- `run_simple`    — one zero-shot solution, scored on the hidden tests (pass@1
                    base; no internal tests, no iteration).
- `run_reflexion` — a single chain: generate, run internal tests, self-reflect,
                    regenerate with accumulated reflection memory, up to
                    `max_iters` times (Shinn et al., 2023). No branching.
- `run_dfs`       — depth-first tree search (Tree-of-Thoughts style): expand `n`
                    children, evaluate on internal tests, recurse into the
                    highest-scoring child first, backtrack to siblings on dead
                    ends. No UCT and no backpropagation — that is what
                    distinguishes it from LATS/MCTS. Bounded by a candidate
                    budget so cost is comparable to LATS.
"""

from __future__ import annotations

import random
from typing import Callable, Optional

from . import executor, generator
from .config import LATSConfig
from .llm import LLM
from .node import Node, branch_history
from .result import ProblemResult


def _hidden_passer(item: dict, config: LATSConfig) -> Callable[[str], bool]:
    def hidden_pass(code: str) -> bool:
        return executor.evaluate(
            item["entry_point"], code, item["test"], config.eval_timeout
        )

    return hidden_pass


def _first_impl(llm: LLM, func_sig: str, config: LATSConfig) -> str:
    for _ in range(3):
        code = generator.simple_impl(llm, func_sig, temperature=config.temperature)
        if code:
            return code
    return ""


# ---------------------------------------------------------------------------
# Simple (pass@1 base)
# ---------------------------------------------------------------------------
def run_simple(
    item: dict,
    llm: LLM,
    config: LATSConfig,
    rng: Optional[random.Random] = None,
    log: Optional[Callable[[str], None]] = None,
) -> ProblemResult:
    say = log or (lambda _m: None)
    code = _first_impl(llm, item["prompt"], config)
    solved = _hidden_passer(item, config)(code)
    say(f"  simple: {'solved ✓' if solved else 'failed'}")
    return ProblemResult(
        name=item["name"],
        solved=solved,
        final_code=code,
        strategy="simple",
        solved_on_first_try=solved,
        iterations_used=0,
        num_candidates=1,
    )


# ---------------------------------------------------------------------------
# Reflexion (linear self-reflection loop)
# ---------------------------------------------------------------------------
def run_reflexion(
    item: dict,
    llm: LLM,
    config: LATSConfig,
    rng: Optional[random.Random] = None,
    log: Optional[Callable[[str], None]] = None,
) -> ProblemResult:
    say = log or (lambda _m: None)
    func_sig = item["prompt"]
    hidden_pass = _hidden_passer(item, config)
    tests = generator.internal_tests(llm, func_sig, config.number_of_tests, rng)

    code = _first_impl(llm, func_sig, config)
    candidates = 1
    impls: list[str] = []
    feedbacks: list[str] = []
    reflections: list[str] = []

    for trial in range(config.max_iters + 1):
        res = executor.execute(code, tests, timeout=config.test_timeout)
        if res.is_passing and hidden_pass(code):
            say(f"  reflexion: solved at trial {trial} ✓")
            return ProblemResult(
                name=item["name"],
                solved=True,
                final_code=code,
                strategy="reflexion",
                solved_on_first_try=(trial == 0),
                iterations_used=trial,
                num_candidates=candidates,
                internal_tests=tests,
            )
        if trial == config.max_iters:
            break
        # Reflect, then regenerate with the full reflection memory so far.
        reflection = generator.self_reflection(llm, code, res.feedback)
        impls.append(code)
        feedbacks.append(res.feedback)
        reflections.append(reflection)
        nxt = generator.reflexion_impls(
            llm,
            func_sig,
            impls,
            feedbacks,
            reflections,
            n=1,
            temperature=config.temperature,
        )
        if not nxt:
            break
        code = nxt[0]
        candidates += 1

    solved = hidden_pass(code)
    say(
        f"  reflexion: {'solved ✓' if solved else 'failed'} after {candidates} candidates"
    )
    return ProblemResult(
        name=item["name"],
        solved=solved,
        final_code=code,
        strategy="reflexion",
        iterations_used=config.max_iters,
        num_candidates=candidates,
        internal_tests=tests,
    )


# ---------------------------------------------------------------------------
# DFS / Tree-of-Thoughts (no UCT, no backprop)
# ---------------------------------------------------------------------------
def run_dfs(
    item: dict,
    llm: LLM,
    config: LATSConfig,
    rng: Optional[random.Random] = None,
    log: Optional[Callable[[str], None]] = None,
) -> ProblemResult:
    say = log or (lambda _m: None)
    func_sig = item["prompt"]
    hidden_pass = _hidden_passer(item, config)
    tests = generator.internal_tests(llm, func_sig, config.number_of_tests, rng)

    root = Node(_first_impl(llm, func_sig, config))
    state = {"candidates": 1, "solved": False, "final": root.solution}
    budget = config.max_iters * config.expansion_factor  # comparable to LATS

    root_res = executor.execute(root.solution, tests, timeout=config.test_timeout)
    root.test_feedback = root_res.feedback
    root.value = root_res.reward
    if root_res.is_passing and hidden_pass(root.solution):
        state["solved"], state["final"] = True, root.solution

    best = {"value": root.value, "code": root.solution}

    def dfs(node: Node, depth: int) -> None:
        if state["solved"] or depth > config.max_iters or state["candidates"] >= budget:
            return
        impls, feedbacks, reflections = branch_history(node)
        children_code = generator.reflexion_impls(
            llm,
            func_sig,
            impls,
            feedbacks,
            reflections,
            n=config.expansion_factor,
            temperature=config.temperature,
        )
        scored: list[Node] = []
        for code in children_code:
            if state["candidates"] >= budget:
                break
            state["candidates"] += 1
            res = executor.execute(code, tests, timeout=config.test_timeout)
            child = Node(code, parent=node, depth=node.depth + 1)
            child.test_feedback = res.feedback
            child.value = res.reward
            if res.reward > best["value"]:
                best["value"], best["code"] = res.reward, code
            if not res.is_passing:
                child.reflection = generator.self_reflection(llm, code, res.feedback)
            elif hidden_pass(code):
                state["solved"], state["final"] = True, code
                return
            scored.append(child)
        # Recurse into the most promising child first (depth-first, value-ordered).
        for child in sorted(scored, key=lambda c: c.value, reverse=True):
            if state["solved"]:
                return
            dfs(child, depth + 1)

    if not state["solved"]:
        dfs(root, 1)

    if not state["solved"]:
        state["final"] = best["code"]
        state["solved"] = hidden_pass(best["code"])

    say(
        f"  dfs: {'solved ✓' if state['solved'] else 'failed'} ({state['candidates']} candidates)"
    )
    return ProblemResult(
        name=item["name"],
        solved=state["solved"],
        final_code=state["final"],
        strategy="dfs",
        num_candidates=state["candidates"],
        internal_tests=tests,
    )


STRATEGIES: dict[str, Callable[..., ProblemResult]] = {
    "simple": run_simple,
    "reflexion": run_reflexion,
    "dfs": run_dfs,
}
