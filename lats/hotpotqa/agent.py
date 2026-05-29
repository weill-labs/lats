"""The LLM-as-(ReAct agent, value function, reflector) wrappers for HotPotQA.

A trajectory is a list of `Step`s. `render_trajectory` serialises it back into the
Thought/Action/Observation text the prompts expect. `generate_steps` samples the
next (thought, action) pairs; `evaluate_state` is the LATS LM value function;
`reflect` explains a failed trajectory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

from ..llm import LLM
from . import prompts

_THOUGHT_RE = re.compile(
    r"Thought\s*\d*\s*:\s*(.*?)\n\s*Action", re.DOTALL | re.IGNORECASE
)
_ACTION_RE = re.compile(r"Action\s*\d*\s*:\s*(.+)", re.IGNORECASE)
_INT_RE = re.compile(r"-?\d+")


@dataclass
class Step:
    thought: str
    action: str  # raw action string, e.g. "Search[Eiffel Tower]"
    observation: str = ""


def render_trajectory(steps: List[Step], include_blank_next: bool = False) -> str:
    """Serialise steps as 'Thought i: ... / Action i: ... / Observation i: ...'."""
    lines: List[str] = []
    for i, s in enumerate(steps, 1):
        lines.append(f"Thought {i}: {s.thought}")
        lines.append(f"Action {i}: {s.action}")
        if s.observation:
            lines.append(f"Observation {i}: {s.observation}")
    if include_blank_next:
        lines.append(f"Thought {len(steps) + 1}:")
    return "\n".join(lines)


def _parse_step(text: str) -> "Tuple[str, str] | None":
    action_m = _ACTION_RE.search(text)
    if not action_m:
        return None
    action = action_m.group(1).strip().splitlines()[0].strip()
    thought_m = _THOUGHT_RE.search(text)
    thought = thought_m.group(1).strip() if thought_m else ""
    return thought, action


def generate_steps(
    llm: LLM,
    question: str,
    steps: List[Step],
    reflections: List[str],
    n: int,
    temperature: float = 0.8,
) -> List[Tuple[str, str]]:
    """Sample `n` candidate next (thought, action) pairs. Drops unparseable ones."""
    traj_text = render_trajectory(steps, include_blank_next=False)
    messages = prompts.react_step_messages(question, traj_text, reflections)
    outs = llm.chat(messages, n=n, temperature=temperature, max_tokens=256)
    parsed = [_parse_step(o) for o in outs]
    return [p for p in parsed if p]


def evaluate_state(llm: LLM, question: str, steps: List[Step]) -> float:
    """LM value function: rate the partial trajectory, normalised to (0, 1]."""
    traj_text = render_trajectory(steps)
    out = llm.chat(
        prompts.value_messages(question, traj_text), n=1, temperature=0.0, max_tokens=8
    )[0]
    m = _INT_RE.search(out)
    if not m:
        return 0.5
    score = max(1, min(10, int(m.group())))
    return score / 10.0


def reflect(llm: LLM, question: str, steps: List[Step]) -> str:
    traj_text = render_trajectory(steps)
    return llm.chat(
        prompts.reflect_messages(question, traj_text),
        n=1,
        temperature=0.0,
        max_tokens=256,
    )[0].strip()
