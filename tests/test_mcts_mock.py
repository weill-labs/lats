"""End-to-end MCTS exercised with a deterministic MockLLM (no API).

Proves the full loop wires together: internal-test generation -> root -> hidden
fast path, and the search path where a buggy root is repaired by a reflexion
expansion that the hidden tests then confirm.
"""

from lats.config import LATSConfig
from lats.llm import MockLLM
from lats.mcts import run_lats_on_problem

PROBLEM = {
    "name": "mock_add",
    "prompt": 'def add(a, b):\n    """Return a + b."""\n',
    "entry_point": "add",
    "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(0, 0) == 0\n",
}

CORRECT = "```python\ndef add(a, b):\n    return a + b\n```"
BUGGY = "```python\ndef add(a, b):\n    return a - b\n```"
TESTS_OUT = "assert add(1, 2) == 3\nassert add(2, 2) == 4"


def _role(messages):
    return messages[0].content


def test_solved_on_first_try():
    def handler(messages, n):
        sys = _role(messages)
        if "[ROLE:tests]" in sys:
            return [TESTS_OUT]
        if "[ROLE:simple]" in sys:
            return [CORRECT]
        return ["unused"] * n

    llm = MockLLM(handler)
    cfg = LATSConfig(max_iters=2, expansion_factor=2, number_of_tests=2)
    result = run_lats_on_problem(PROBLEM, llm, cfg)

    assert result.solved
    assert result.solved_on_first_try
    assert result.iterations_used == 0
    assert "return a + b" in result.final_code
    assert llm.num_calls >= 2  # tests + simple


def test_solved_after_reflexion():
    def handler(messages, n):
        sys = _role(messages)
        if "[ROLE:tests]" in sys:
            return [TESTS_OUT]
        if "[ROLE:simple]" in sys:
            return [BUGGY]  # root fails internal tests
        if "[ROLE:reflect]" in sys:
            return ["It subtracts instead of adding."] * n
        if "[ROLE:reflexion]" in sys:
            return [CORRECT] * n  # expansion repairs it
        return ["unused"] * n

    llm = MockLLM(handler)
    cfg = LATSConfig(max_iters=2, expansion_factor=2, number_of_tests=2)
    result = run_lats_on_problem(PROBLEM, llm, cfg)

    assert result.solved
    assert not result.solved_on_first_try
    assert result.iterations_used >= 1
    assert result.num_candidates >= 2  # root + at least one child
    assert "return a + b" in result.final_code


def test_unsolved_when_only_buggy():
    def handler(messages, n):
        sys = _role(messages)
        if "[ROLE:tests]" in sys:
            return [TESTS_OUT]
        if "[ROLE:reflect]" in sys:
            return ["wrong"] * n
        if "[ROLE:reflexion]" in sys:
            return [BUGGY] * n
        return [BUGGY] * n  # simple also buggy

    llm = MockLLM(handler)
    cfg = LATSConfig(max_iters=2, expansion_factor=2, number_of_tests=2)
    result = run_lats_on_problem(PROBLEM, llm, cfg)

    assert not result.solved
