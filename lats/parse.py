"""Extract a Python function body from an LLM response.

LLMs wrap code in ```python ... ``` fences most of the time, but not always.
We try, in order: a language-tagged fence, a bare fence, then a heuristic that
grabs the first top-level `def ...` block. Mirrors the official LATS parser but
written from scratch and kept small.
"""

from __future__ import annotations

import re
from typing import Optional

_FENCE_LANG = re.compile(r"```[a-zA-Z]*\n(.*?)\n```", re.DOTALL)


def add_code_block(code: str, lang: str = "python") -> str:
    """Wrap `code` in a fenced block (used when feeding prior impls back in)."""
    return f"```{lang}\n{code}\n```"


def parse_code_block(text: str, lang: str = "python") -> Optional[str]:
    """Return the code inside the first fenced block, or fall back to a def-scan.

    Returns None only when no function-looking code can be found at all.
    """
    # Prefer a fence whose tag matches the language, then any fence.
    lang_fence = re.search(rf"```{lang}\n(.*?)\n```", text, re.DOTALL)
    if lang_fence:
        return lang_fence.group(1).strip("\n")

    any_fence = _FENCE_LANG.search(text)
    if any_fence:
        return any_fence.group(1).strip("\n")

    return _parse_first_func(text)


def _parse_first_func(text: str) -> Optional[str]:
    """Heuristic: keep from the first top-level `def` to the end of its block.

    Stops at the next top-level `def`/`class` or a blank line that follows a
    `return`, whichever comes first. Good enough for single-function HumanEval
    answers when the model forgot the code fence.
    """
    lines = text.split("\n")
    start = -1
    saw_return = False
    end = len(lines)

    for i, line in enumerate(lines):
        if line.startswith("def "):
            if start == -1:
                start = i
            else:
                end = i
                break
        elif start != -1 and line.startswith("class "):
            end = i
            break
        elif start != -1 and "return" in line:
            saw_return = True
        elif start != -1 and saw_return and line.strip() == "":
            end = i
            break

    if start == -1:
        return None
    return "\n".join(lines[start:end]).rstrip()
