"""Prompt templates for the four LLM roles in LATS-for-code.

These instructions and few-shots follow the official LATS `programming/`
implementation (which itself builds on Reflexion) so behaviour stays faithful to
the paper. Each builder returns a `list[Message]` ready for `LLM.chat`. A short
ROLE tag is embedded in every system prompt so `MockLLM` handlers (and humans
reading logs) can tell the four call types apart.
"""

from __future__ import annotations

from typing import List

from .llm import Message
from .parse import add_code_block

CODE_BLOCK_INSTRUCTION = (
    "Use a Python code block to write your response. For example:\n"
    "```python\nprint('Hello world!')\n```"
)

# --- 1. Initial ("simple") implementation -------------------------------------
SIMPLE_SYSTEM = (
    "[ROLE:simple] You are an AI that only responds with python code, NOT "
    "ENGLISH. You will be given a function signature and its docstring by the "
    "user. Write your full implementation (restate the function signature)."
)

# --- 2. Reflexion implementation (used during MCTS expansion) ------------------
REFLEXION_SYSTEM = (
    "[ROLE:reflexion] You are an AI Python assistant. You will be given your "
    "previous function implementations, the unit test results for each, and your "
    "self-reflections. Use them to write an improved implementation (restate the "
    "function signature)."
)

REFLEXION_FEW_SHOT = '''Example 1:
[previous impl]:
```python
def add(a: int, b: int) -> int:
    """ Given integers a and b, return the total value of a and b. """
    return a - b
```

[unit test results from previous impl]:
Tests passed:

Tests failed:
assert add(1, 2) == 3 # output: -1

[reflection on previous impl]:
The implementation is wrong because it subtracts instead of adding. Change the
`-` operator to `+` in the return statement.

[improved impl]:
```python
def add(a: int, b: int) -> int:
    """ Given integers a and b, return the total value of a and b. """
    return a + b
```
'''

# --- 3. Self-reflection -------------------------------------------------------
SELF_REFLECTION_SYSTEM = (
    "[ROLE:reflect] You are a Python programming assistant. You will be given a "
    "function implementation and a series of unit test results. Write a few "
    "sentences explaining why the implementation is wrong as indicated by the "
    "tests. You will use this as a hint when you try again later. Provide only "
    "the few-sentence description, not the implementation."
)

SELF_REFLECTION_FEW_SHOT = """Example 1:
[function impl]:
```python
def longest_subarray_with_sum_limit(nums: List[int], target: int) -> List[int]:
    n = len(nums); left = right = 0; max_length = 0; current_sum = 0; result = []
    while right < n:
        current_sum += nums[right]
        while current_sum > target:
            current_sum -= nums[left]; left += 1
        if right - left + 1 >= max_length:
            max_length = right - left + 1; result = nums[left:right+1]
        right += 1
    return result
```
[unit test results]:
Tests passed:
assert longest_subarray_with_sum_limit([1, 2, 3, 4, 5], 8) == [1, 2, 3]
Tests failed:
assert longest_subarray_with_sum_limit([5, 6, 7, 8, 9], 4) == [] # output: [5]
[self-reflection]:
The code uses `>=` instead of `>` when updating the result, so it returns a
subarray even when none satisfies the constraint. Use `>` so the result only
updates on a strictly longer valid subarray.
END OF EXAMPLES
"""

# --- 4. Internal test generation ---------------------------------------------
TEST_GEN_SYSTEM = (
    "[ROLE:tests] You are an AI coding assistant that writes unique, diverse, "
    "and intuitive unit tests for functions given the signature and docstring. "
    "Respond with assert statements only."
)

TEST_GEN_FEW_SHOT = '''func signature:
def add3Numbers(x, y, z):
    """ Add three numbers together and return the sum. """
unit tests:
assert add3Numbers(1, 2, 3) == 6
assert add3Numbers(-1, 2, 3) == 4
assert add3Numbers(1, -2, 3) == 2
assert add3Numbers(1, 2, -3) == 0
assert add3Numbers(-3, -2, -1) == -6
assert add3Numbers(0, 0, 0) == 0
'''


def simple_impl_messages(func_sig: str) -> List[Message]:
    return [
        Message("system", f"{SIMPLE_SYSTEM}\n{CODE_BLOCK_INSTRUCTION}"),
        Message("user", func_sig),
    ]


def reflexion_impl_messages(
    func_sig: str,
    prev_impls: List[str],
    feedbacks: List[str],
    reflections: List[str],
) -> List[Message]:
    """Multi-turn prompt that replays every prior attempt on this branch.

    Layout: system + few-shot, then for each ancestor an assistant turn (its
    code) followed by a user turn (its test results + reflection), and finally a
    user turn asking for the improved implementation.
    """
    messages = [
        Message("system", f"{REFLEXION_SYSTEM}\n{CODE_BLOCK_INSTRUCTION}"),
        Message("user", REFLEXION_FEW_SHOT),
    ]
    for impl, feedback, reflection in zip(prev_impls, feedbacks, reflections):
        messages.append(Message("assistant", add_code_block(impl)))
        messages.append(
            Message(
                "user",
                f"[unit test results from previous impl]:\n{feedback}\n\n"
                f"[reflection on previous impl]:\n{reflection}",
            )
        )
    messages.append(Message("user", f"[improved impl]:\n{func_sig}"))
    return messages


def self_reflection_messages(func: str, feedback: str) -> List[Message]:
    return [
        Message("system", SELF_REFLECTION_SYSTEM),
        Message(
            "user",
            f"{SELF_REFLECTION_FEW_SHOT}\n\n[function impl]:\n{add_code_block(func)}\n\n"
            f"[unit test results]:\n{feedback}\n\n[self-reflection]:",
        ),
    ]


def test_generation_messages(func_sig: str) -> List[Message]:
    return [
        Message("system", f"{TEST_GEN_SYSTEM}\n\n{TEST_GEN_FEW_SHOT}"),
        Message("user", f"[func signature]:\n{func_sig}\n\n[unit tests]:"),
    ]
