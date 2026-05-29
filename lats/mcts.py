"""LATS search for one HumanEval problem.

This is the clean-room core. It implements the MCTS loop the paper describes,
specialised to code where the value signal is "fraction of self-generated tests
passed":

    1. internal tests   — LLM writes asserts (the hidden tests are unseen)
    2. root             — an initial "simple" solution; if it already passes the
                          hidden tests we're done (the pass@1 fast path)
    3. for each iteration:
         selection      — descend from root by UCT to a leaf
         expansion      — sample `n` children conditioned on the branch's history
         simulation     — run each child on the internal tests -> reward
         (hidden check) — child passing internal tests (or last iter) is scored
                          on the hidden tests; a real pass ends the search
         backprop       — add reward to the child and all ancestors

We deliberately diverge from the official `mcts.py`, whose expansion loop
re-iterates over `node.children` with an early `break` and backpropagates a
dangling variable. Reward semantics (`internal_fraction + real_pass`) are kept.
"""

from __future__ import annotations

import random
from typing import Callable, Optional

from . import executor, generator
from .config import LATSConfig
from .llm import LLM
from .node import Node, branch_history
from .result import ProblemResult


def run_lats_on_problem(
    item: dict,
    llm: LLM,
    config: LATSConfig,
    rng: Optional[random.Random] = None,
    log: Optional[Callable[[str], None]] = None,
) -> ProblemResult:
    say = log or (lambda _msg: None)
    func_sig = item["prompt"]
    entry_point = item["entry_point"]
    hidden_test = item["test"]

    def hidden_pass(code: str) -> bool:
        return executor.evaluate(entry_point, code, hidden_test, config.eval_timeout)

    # 1. Self-generated internal tests (the reward signal).
    tests = generator.internal_tests(llm, func_sig, config.number_of_tests, rng)
    say(f"  internal tests: {len(tests)}")

    # 2. Initial "simple" solution -> root.
    root_code = None
    for _ in range(3):
        root_code = generator.simple_impl(llm, func_sig, temperature=config.temperature)
        if root_code:
            break
    root = Node(root_code or "")
    num_candidates = 1

    res = executor.execute(root.solution, tests, timeout=config.test_timeout)
    root.test_feedback = res.feedback
    root.update(res.reward)  # seed root with its own value/visit

    # Fast path: root already passes the hidden tests (paper's pass@1 case).
    if res.is_passing and hidden_pass(root.solution):
        say("  solved on first try ✓")
        return ProblemResult(
            name=item["name"],
            solved=True,
            final_code=root.solution,
            strategy="lats",
            solved_on_first_try=True,
            iterations_used=0,
            num_candidates=num_candidates,
            internal_tests=tests,
        )

    # Reflect on the root so its children can learn from it.
    if not res.is_passing:
        root.reflection = generator.self_reflection(llm, root.solution, res.feedback)

    solved = False
    final_code = root.solution
    iterations_used = 0

    for it in range(config.max_iters):
        iterations_used = it + 1
        last_iter = it == config.max_iters - 1
        # Fresh internal tests each iteration (matches official behaviour; adds
        # diversity to the reward signal so we don't overfit one test set).
        tests = generator.internal_tests(llm, func_sig, config.number_of_tests, rng)

        # --- Selection: descend to a leaf by UCT ---
        node = root
        while node.children:
            node = node.best_child(config.exploration_weight)

        # --- Expansion: sample n children from the branch's accumulated history ---
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
        if not children_code:  # all completions unparseable; try one plain sample
            fallback = generator.simple_impl(
                llm, func_sig, temperature=config.temperature
            )
            children_code = [fallback] if fallback else []

        for code in children_code:
            child = Node(code, parent=node, depth=node.depth + 1)
            node.children.append(child)
            num_candidates += 1

            # --- Simulation: evaluate on internal tests ---
            res = executor.execute(code, tests, timeout=config.test_timeout)
            child.test_feedback = res.feedback
            reward = res.reward
            if not res.is_passing:
                child.reflection = generator.self_reflection(llm, code, res.feedback)

            # --- Hidden-test check (final scorer) ---
            reward_real = 0.0
            if res.is_passing or last_iter:
                if hidden_pass(code):
                    solved, final_code, reward_real = True, code, 1.0

            # --- Backpropagation ---
            child.backpropagate(reward + reward_real)

            if solved:
                say(f"  solved at iter {iterations_used} ✓")
                break
        if solved:
            break

    # Choose the reported answer: the solver if found, else the highest-mean child.
    if not solved:
        best = root.best_child_value() or root
        final_code = best.solution
        solved = hidden_pass(final_code)
        say(f"  best candidate {'passed' if solved else 'failed'} hidden tests")

    return ProblemResult(
        name=item["name"],
        solved=solved,
        final_code=final_code,
        solved_on_first_try=False,
        iterations_used=iterations_used,
        num_candidates=num_candidates,
        internal_tests=tests,
    )
