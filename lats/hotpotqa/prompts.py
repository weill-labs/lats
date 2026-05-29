"""Prompts for the HotPotQA domain: ReAct agent, LM value function, reflection.

The agent prompt is the standard ReAct format (Thought/Action interleaved with
Observations); the value function is the LATS LM scorer that rates how promising
a partial trajectory is; reflection explains a failed trajectory so later
expansions avoid the same mistake. As with the HumanEval prompts, each system
message carries a [ROLE:*] tag so MockLLM handlers can route deterministically.
"""

from __future__ import annotations

from typing import List

from ..llm import Message

REACT_SYSTEM = (
    "[ROLE:react] Solve a question-answering task with interleaving Thought, "
    "Action, Observation steps. Thought reasons about the current situation. "
    "Action must be one of:\n"
    "  Search[entity] — look up a Wikipedia page and read its first paragraph\n"
    "  Lookup[keyword] — return the next sentence on the current page with keyword\n"
    "  Finish[answer] — return the answer and finish the task\n"
    "Respond with exactly one Thought and one Action, formatted as:\n"
    "Thought: <your reasoning>\nAction: <Search[...]|Lookup[...]|Finish[...]>"
)

REACT_FEW_SHOT = """Question: What profession do Nicholas Ray and Elia Kazan have in common?
Thought: I need to find the professions of Nicholas Ray and Elia Kazan, then find the overlap.
Action: Search[Nicholas Ray]
Observation: Nicholas Ray (1911-1979) was an American film director, screenwriter, and actor.
Thought: Nicholas Ray was a director, screenwriter, and actor. Now I need Elia Kazan's professions.
Action: Search[Elia Kazan]
Observation: Elia Kazan was an American film and theatre director, producer, screenwriter and actor.
Thought: Both were directors, screenwriters, and actors. The common professions are director, screenwriter, and actor.
Action: Finish[director, screenwriter, actor]"""

VALUE_SYSTEM = (
    "[ROLE:value] You evaluate how promising a partial reasoning trajectory is "
    "for answering a question. Consider whether the steps are relevant, factually "
    "grounded, and progressing toward a correct answer. Respond with a single "
    "integer score from 1 (hopeless) to 10 (almost certainly leads to the correct "
    "answer)."
)

REFLECT_SYSTEM = (
    "[ROLE:reflect-qa] You are given a question and a failed reasoning trajectory. "
    "Write a few sentences diagnosing why it failed and what to do differently next "
    "time (e.g. a better entity to search, a missed clue). Provide only the short "
    "reflection."
)


def react_step_messages(
    question: str, trajectory_text: str, reflections: List[str]
) -> List[Message]:
    memory = ""
    if reflections:
        joined = "\n".join(f"- {r}" for r in reflections)
        memory = f"\n\nReflections from previous failed attempts:\n{joined}"
    user = (
        f"{REACT_FEW_SHOT}\n\n(Now solve this one.){memory}\n\n"
        f"Question: {question}\n{trajectory_text}"
    ).rstrip()
    return [Message("system", REACT_SYSTEM), Message("user", user)]


def value_messages(question: str, trajectory_text: str) -> List[Message]:
    user = (
        f"Question: {question}\n{trajectory_text}\n\n"
        "How promising is this trajectory? Answer with a single integer 1-10."
    )
    return [Message("system", VALUE_SYSTEM), Message("user", user)]


def reflect_messages(question: str, trajectory_text: str) -> List[Message]:
    user = f"Question: {question}\n{trajectory_text}\n\nReflection:"
    return [Message("system", REFLECT_SYSTEM), Message("user", user)]
