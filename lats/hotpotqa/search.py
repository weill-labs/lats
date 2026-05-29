"""LATS search over ReAct trajectories for one HotPotQA question.

Each tree node is one ReAct step (Thought + Action + Observation); the path from
the root to a node is a trajectory. We reuse `lats.node.Node` (UCT + backprop)
and add the QA specifics:

- expansion samples `n` next steps and *executes* each action in a Wikipedia env
  replayed to the parent's state (fetchers are cached, so replay is cheap);
- a terminal `Finish[answer]` node gets the ground-truth reward (1.0 if the answer
  exact-matches the gold, else 0.0 + a reflection that feeds later expansions);
- a non-terminal node gets the LM value-function estimate as its reward.

Selection is best-first over the live frontier of expandable (non-terminal,
depth-bounded) leaves, scored by UCT — a pragmatic MCTS variant (documented in
docs/DESIGN.md) rather than strict root-to-leaf descent.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, List, Optional

from ..config import LATSConfig
from ..llm import LLM
from ..node import Node
from . import wikienv
from .agent import Step, evaluate_state, generate_steps, reflect
from .scoring import exact_match, f1_score


@dataclass
class HotpotResult:
    id: str
    question: str
    gold: str
    predicted: Optional[str]
    em: bool
    f1: float
    solved: bool
    iterations_used: int
    num_nodes: int


def _steps_to(node: Node) -> List[Step]:
    chain: List[Step] = []
    cur: Optional[Node] = node
    while cur is not None and getattr(cur, "step", None) is not None:
        chain.append(cur.step)
        cur = cur.parent
    chain.reverse()
    return chain


def run_lats_hotpotqa(
    item: dict,
    llm: LLM,
    config: LATSConfig,
    get_page: Optional[Callable[[str], Optional[str]]] = None,
    search_titles: Optional[Callable[[str], List[str]]] = None,
    rng: Optional[random.Random] = None,
    log: Optional[Callable[[str], None]] = None,
) -> HotpotResult:
    say = log or (lambda _m: None)
    question, gold = item["question"], item["answer"]

    # Cache the (possibly networked) fetchers so replaying a branch is cheap.
    base_get = get_page or wikienv._mediawiki_get_page
    base_search = search_titles or wikienv._mediawiki_search
    page_cache: dict = {}
    search_cache: dict = {}

    def cached_get(t: str):
        if t not in page_cache:
            page_cache[t] = base_get(t)
        return page_cache[t]

    def cached_search(q: str):
        if q not in search_cache:
            search_cache[q] = base_search(q)
        return search_cache[q]

    def env_at(node: Node) -> wikienv.WikipediaEnv:
        env = wikienv.WikipediaEnv(get_page=cached_get, search_titles=cached_search)
        for s in _steps_to(node):
            pa = wikienv.parse_action(s.action)
            if pa:
                env.act(*pa)
        return env

    root = Node("")
    root.step = None
    root.is_terminal = False
    num_nodes = 1
    reflections: List[str] = []
    best = {"answer": None, "em": False, "f1": -1.0}
    solved = False
    iterations = 0

    def frontier() -> List[Node]:
        out, stack = [], [root]
        while stack:
            nd = stack.pop()
            if nd.children:
                stack.extend(nd.children)
            elif (
                not getattr(nd, "is_terminal", False)
                and nd.depth < config.react_max_depth
            ):
                out.append(nd)
        return out

    for it in range(config.max_iters):
        iterations = it + 1
        fr = frontier()
        if not fr:
            break
        node = max(
            fr,
            key=lambda nd: (
                float("inf") if nd.parent is None else nd.uct(config.exploration_weight)
            ),
        )
        steps = _steps_to(node)
        candidates = generate_steps(
            llm,
            question,
            steps,
            reflections,
            config.expansion_factor,
            config.temperature,
        )
        if not candidates:
            node.is_terminal = True  # exhausted/dead-end; drop from frontier
            continue

        for thought, action in candidates:
            env = env_at(node)
            pa = wikienv.parse_action(action)
            obs = env.act(*pa) if pa else f"Invalid action: {action}"
            step = Step(thought=thought, action=action, observation=obs)
            child = Node(action, parent=node, depth=node.depth + 1)
            child.step = step
            child.is_terminal = env.done
            node.children.append(child)
            num_nodes += 1
            full = steps + [step]

            if env.done:
                ans = env.answer or ""
                em, f1 = exact_match(ans, gold), f1_score(ans, gold)
                reward = 1.0 if em else 0.0
                if (em and not best["em"]) or (em == best["em"] and f1 > best["f1"]):
                    best = {"answer": ans, "em": em, "f1": f1}
                if em:
                    solved = True
                else:
                    reflections.append(reflect(llm, question, full))
            else:
                reward = evaluate_state(llm, question, full)

            child.backpropagate(reward)
            if solved:
                break
        if solved:
            break

    em = best["em"]
    f1 = best["f1"] if best["f1"] >= 0 else 0.0
    say(
        f"  hotpotqa: {'EM ✓' if em else 'EM ✗'} pred={best['answer']!r} gold={gold!r} "
        f"({iterations} iters, {num_nodes} nodes)"
    )
    return HotpotResult(
        id=str(item.get("id", "")),
        question=question,
        gold=gold,
        predicted=best["answer"],
        em=em,
        f1=f1,
        solved=em,
        iterations_used=iterations,
        num_nodes=num_nodes,
    )
