"""Baseline strategies exercised with a deterministic MockLLM (no API)."""

from lats.config import LATSConfig
from lats.llm import MockLLM
from lats.strategies import run_dfs, run_reflexion, run_simple

PROBLEM = {
    "name": "mock_add",
    "prompt": 'def add(a, b):\n    """Return a + b."""\n',
    "entry_point": "add",
    "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(0, 0) == 0\n",
}
CORRECT = "```python\ndef add(a, b):\n    return a + b\n```"
BUGGY = "```python\ndef add(a, b):\n    return a - b\n```"
TESTS_OUT = "assert add(1, 2) == 3\nassert add(2, 2) == 4"
CFG = LATSConfig(max_iters=2, expansion_factor=2, number_of_tests=2)


def _handler(simple_ret, reflexion_ret):
    def handler(messages, n):
        sys = messages[0].content
        if "[ROLE:tests]" in sys:
            return [TESTS_OUT]
        if "[ROLE:reflect]" in sys:
            return ["reflection"] * n
        if "[ROLE:reflexion]" in sys:
            return [reflexion_ret] * n
        return [simple_ret] * n  # [ROLE:simple]

    return handler


def test_simple_solved_and_unsolved():
    assert run_simple(PROBLEM, MockLLM(_handler(CORRECT, CORRECT)), CFG).solved
    r = run_simple(PROBLEM, MockLLM(_handler(BUGGY, BUGGY)), CFG)
    assert not r.solved
    assert r.strategy == "simple"
    assert r.num_candidates == 1


def test_reflexion_repairs_buggy_root():
    r = run_reflexion(PROBLEM, MockLLM(_handler(BUGGY, CORRECT)), CFG)
    assert r.solved
    assert r.strategy == "reflexion"
    assert r.num_candidates >= 2  # root + at least one regeneration
    assert "return a + b" in r.final_code


def test_reflexion_unsolved_when_always_buggy():
    assert not run_reflexion(PROBLEM, MockLLM(_handler(BUGGY, BUGGY)), CFG).solved


def test_dfs_finds_correct_child():
    r = run_dfs(PROBLEM, MockLLM(_handler(BUGGY, CORRECT)), CFG)
    assert r.solved
    assert r.strategy == "dfs"
    assert r.num_candidates >= 2


def test_dfs_respects_candidate_budget_when_unsolved():
    r = run_dfs(PROBLEM, MockLLM(_handler(BUGGY, BUGGY)), CFG)
    assert not r.solved
    # budget = max_iters * expansion_factor = 4; root + children must not exceed it by much
    assert r.num_candidates <= CFG.max_iters * CFG.expansion_factor + 1
