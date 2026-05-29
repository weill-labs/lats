"""The MCTS search node and tree-context gathering.

A node holds one candidate solution plus the test feedback and self-reflection
that explain how it did. `value` is the *sum* of rewards backpropagated through
it and `visits` the number of those updates, so the mean reward is
`value / visits` — exactly the exploitation term of UCT.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple


class Node:
    def __init__(self, solution: str, parent: "Optional[Node]" = None, depth: int = 0):
        self.solution = solution
        self.parent = parent
        self.children: List["Node"] = []
        self.value: float = 0.0
        self.visits: int = 0
        self.depth = depth
        self.reflection: str = ""  # LM self-reflection on this solution's failures
        self.test_feedback: str = ""  # internal-test results string for this solution

    # --- UCT selection ---------------------------------------------------
    def uct(self, exploration_weight: float = 1.0) -> float:
        """Upper Confidence bound for Trees.

        Unvisited nodes return +inf so selection always expands them first
        (standard MCTS). This is a deliberate fix over the official code, which
        returned the raw value for unvisited nodes.
        """
        if self.visits == 0:
            return float("inf")
        exploit = self.value / self.visits
        explore = exploration_weight * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploit + explore

    def best_child(self, exploration_weight: float = 1.0) -> "Optional[Node]":
        """Child with the highest UCT score (used during selection)."""
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.uct(exploration_weight))

    def best_child_value(self) -> "Optional[Node]":
        """Child with the highest mean reward (used to pick the final answer)."""
        if not self.children:
            return None
        return max(
            self.children,
            key=lambda c: (c.value / c.visits) if c.visits else c.value,
        )

    def update(self, reward: float) -> None:
        self.visits += 1
        self.value += reward

    def backpropagate(self, reward: float) -> None:
        """Add `reward` to this node and every ancestor up to the root."""
        node: Optional[Node] = self
        while node is not None:
            node.update(reward)
            node = node.parent

    def __repr__(self) -> str:
        return f"Node(depth={self.depth}, visits={self.visits}, value={self.value:.2f})"


def gather_context_from_tree(node: Node) -> Tuple[List[str], List[str]]:
    """Walk from `node` up to the root, collecting feedback and reflections.

    Returned in root-first order so the prompt reads chronologically (oldest
    attempt first). This is the accumulated context that lets a new expansion
    learn from every failure along its branch.
    """
    feedbacks: List[str] = []
    reflections: List[str] = []
    cur: Optional[Node] = node
    while cur is not None:
        if cur.test_feedback:
            feedbacks.append(cur.test_feedback)
        if cur.reflection:
            reflections.append(cur.reflection)
        cur = cur.parent
    return feedbacks[::-1], reflections[::-1]
