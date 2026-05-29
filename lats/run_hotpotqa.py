"""CLI: run LATS (ReAct + MCTS) over a HotPotQA subset and report EM/F1.

Examples:
    uv run lats-hotpotqa --num-problems 5 --model gpt-3.5-turbo
    uv run lats-hotpotqa --mock --num-problems 2          # no API, plumbing only
    uv run lats-hotpotqa --dataset hotpot_dev_distractor_v1.json --num-problems 50
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from .config import LATSConfig
from .hotpotqa.dataset import load_hotpotqa, select_subset
from .hotpotqa.search import run_lats_hotpotqa
from .llm import LLM, Message, MockLLM, OpenAIChat


def _mock_handler(messages: List[Message], n: int) -> List[str]:
    """No-API stand-in: search once, then Finish (with a placeholder answer).

    Runs the full ReAct+MCTS loop without network/LLM; EM will be ~0 since the
    answer is a placeholder. Plumbing check only — real correctness lives in
    tests/test_hotpotqa_search_mock.py.
    """
    system, user = messages[0].content, messages[1].content
    if "[ROLE:value]" in system:
        return ["5"]
    if "[ROLE:reflect-qa]" in system:
        return ["reconsider the entity to search"]
    if "Observation" in user:  # we've already acted; wrap up
        return ["Thought: I'll answer now.\nAction: Finish[unknown]"] * n
    return ["Thought: Search the main entity.\nAction: Search[Albert Einstein]"] * n


def build_llm(args) -> LLM:
    return MockLLM(_mock_handler) if args.mock else OpenAIChat(args.model)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run LATS (ReAct+MCTS) on HotPotQA.")
    p.add_argument("--model", default="gpt-3.5-turbo")
    p.add_argument("--max-iters", type=int, default=None, help="MCTS iterations")
    p.add_argument(
        "--expansion-factor",
        type=int,
        default=None,
        help="steps sampled per expansion (n)",
    )
    p.add_argument(
        "--react-max-depth",
        type=int,
        default=None,
        help="max ReAct steps per trajectory",
    )
    p.add_argument(
        "--exploration-weight", type=float, default=None, help="UCT constant c"
    )
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--num-problems", type=int, default=5)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--ids", nargs="*", default=None, help="run specific question ids")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--output",
        default=None,
        help="jsonl log path (default logs/hotpotqa_<ts>.jsonl)",
    )
    p.add_argument("--mock", action="store_true", help="use MockLLM (no API calls)")
    p.add_argument(
        "--dataset",
        default=None,
        help="path to HotPotQA json/jsonl (default: vendored sample)",
    )
    return p.parse_args(argv)


def make_config(args) -> LATSConfig:
    cfg = LATSConfig(
        model=args.model, max_iters=6, expansion_factor=3, react_max_depth=7
    )
    if args.max_iters is not None:
        cfg.max_iters = args.max_iters
    if args.expansion_factor is not None:
        cfg.expansion_factor = args.expansion_factor
    if args.react_max_depth is not None:
        cfg.react_max_depth = args.react_max_depth
    if args.exploration_weight is not None:
        cfg.exploration_weight = args.exploration_weight
    if args.temperature is not None:
        cfg.temperature = args.temperature
    return cfg


def main(argv=None) -> int:
    load_dotenv()
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    args = parse_args(argv)
    cfg = make_config(args)
    llm = build_llm(args)
    rng = random.Random(args.seed)

    data = load_hotpotqa(args.dataset)
    problems = select_subset(
        data, num_problems=args.num_problems, start=args.start, ids=args.ids
    )

    out_path = (
        Path(args.output)
        if args.output
        else Path("logs") / f"hotpotqa_{int(time.time())}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"HotPotQA | model={cfg.model} iters={cfg.max_iters} n={cfg.expansion_factor} "
        f"depth={cfg.react_max_depth} | {len(problems)} questions "
        f"{'[MOCK]' if args.mock else ''}"
    )
    print(f"logging to {out_path}\n")

    em_sum = 0.0
    f1_sum = 0.0
    for i, item in enumerate(problems, 1):
        print(f"[{i}/{len(problems)}] {item['question']}")
        r = run_lats_hotpotqa(item, llm, cfg, rng=rng, log=print)
        em_sum += int(r.em)
        f1_sum += r.f1
        with open(out_path, "a") as f:
            f.write(
                json.dumps(
                    {
                        "id": r.id,
                        "question": r.question,
                        "gold": r.gold,
                        "predicted": r.predicted,
                        "em": r.em,
                        "f1": round(r.f1, 3),
                        "iterations_used": r.iterations_used,
                        "num_nodes": r.num_nodes,
                    }
                )
                + "\n"
            )
        print(
            f"    running EM = {em_sum}/{i} = {em_sum / i:.3f}  F1 = {f1_sum / i:.3f}\n"
        )

    n = max(len(problems), 1)
    print("=" * 56)
    print(
        f"EM = {em_sum:.0f}/{len(problems)} = {em_sum / n:.3f}   avg F1 = {f1_sum / n:.3f}"
    )
    print(f"LLM calls: {llm.num_calls}  completions: {llm.num_completions}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
