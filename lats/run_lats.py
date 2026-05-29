"""CLI: run a search strategy over a HumanEval subset and report pass@1.

Examples:
    uv run lats --num-problems 20 --model gpt-3.5-turbo            # LATS (default)
    uv run lats --strategy reflexion --num-problems 10             # one baseline
    uv run lats --strategy all --num-problems 8                    # head-to-head table
    uv run lats --mock --num-problems 3                            # no API, plumbing only
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Callable, List

from dotenv import load_dotenv

from . import strategies
from .config import LATSConfig
from .dataset import load_dataset, select_subset
from .llm import LLM, Message, MockLLM, OpenAIChat
from .mcts import run_lats_on_problem
from .result import ProblemResult

# All selectable strategies. "lats" is our method; the rest are baselines.
STRATEGY_FNS: dict[str, Callable[..., ProblemResult]] = {
    "lats": run_lats_on_problem,
    **strategies.STRATEGIES,
}
# Order used by --strategy all: cheap -> expensive.
ALL_ORDER = ["simple", "reflexion", "dfs", "lats"]


def _mock_handler(messages: List[Message], n: int) -> List[str]:
    """A no-API stand-in: routes on the [ROLE:*] tag in the system prompt.

    Returns a syntactically valid stub that won't match the entry point, so the
    loop runs to completion (exercising expansion/reflection/selection) but
    reports not-solved. This is a plumbing check, not a correctness check — the
    real end-to-end correctness proof lives in tests/test_mcts_mock.py.
    """
    system = messages[0].content
    if "[ROLE:tests]" in system:
        return ["assert True"]
    if "[ROLE:reflect]" in system:
        return ["stub reflection"] * n
    stub = "```python\ndef _stub():\n    return None\n```"
    return [stub] * n


def build_llm(args) -> LLM:
    if args.mock:
        return MockLLM(_mock_handler)
    return OpenAIChat(args.model)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run LATS / baselines on HumanEval.")
    p.add_argument(
        "--strategy",
        default="lats",
        choices=["lats", "simple", "reflexion", "dfs", "all"],
        help="search strategy ('all' runs every strategy and prints a comparison)",
    )
    p.add_argument("--model", default="gpt-3.5-turbo")
    p.add_argument(
        "--max-iters", type=int, default=None, help="iterations/depth (default: preset)"
    )
    p.add_argument(
        "--expansion-factor", type=int, default=None, help="children per expansion (n)"
    )
    p.add_argument(
        "--number-of-tests", type=int, default=None, help="internal tests generated"
    )
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument(
        "--exploration-weight", type=float, default=None, help="UCT constant c"
    )
    p.add_argument("--num-problems", type=int, default=5)
    p.add_argument("--start", type=int, default=0)
    p.add_argument(
        "--names", nargs="*", default=None, help="run specific problem names"
    )
    p.add_argument(
        "--shuffle-seed", type=int, default=None, help="seeded random subset"
    )
    p.add_argument(
        "--seed", type=int, default=0, help="rng seed for internal-test sampling"
    )
    p.add_argument(
        "--output", default=None, help="jsonl log path (default logs/<ts>.jsonl)"
    )
    p.add_argument("--mock", action="store_true", help="use MockLLM (no API calls)")
    p.add_argument("--dataset", default=None, help="override dataset path")
    return p.parse_args(argv)


def make_config(args) -> LATSConfig:
    cfg = LATSConfig.gpt4() if args.model.startswith("gpt-4") else LATSConfig.gpt35()
    cfg.model = args.model
    if args.max_iters is not None:
        cfg.max_iters = args.max_iters
    if args.expansion_factor is not None:
        cfg.expansion_factor = args.expansion_factor
    if args.number_of_tests is not None:
        cfg.number_of_tests = args.number_of_tests
    if args.temperature is not None:
        cfg.temperature = args.temperature
    if args.exploration_weight is not None:
        cfg.exploration_weight = args.exploration_weight
    return cfg


def main(argv=None) -> int:
    # Load OPENAI_API_KEY from .env regardless of the current working directory:
    # first the nearest .env, then the repo-root .env next to this package.
    load_dotenv()
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    args = parse_args(argv)
    cfg = make_config(args)
    llm = build_llm(args)
    rng = random.Random(args.seed)

    data = load_dataset(args.dataset)
    problems = select_subset(
        data,
        num_problems=args.num_problems,
        start=args.start,
        names=args.names,
        shuffle_seed=args.shuffle_seed,
    )

    out_path = (
        Path(args.output)
        if args.output
        else Path("logs") / f"lats_{int(time.time())}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    names = ALL_ORDER if args.strategy == "all" else [args.strategy]
    # Per-strategy tallies; calls are attributed by snapshotting the LLM counter.
    tally = {n: {"solved": 0, "candidates": 0, "calls": 0} for n in names}

    print(
        f"strategies={names} | model={cfg.model} iters={cfg.max_iters} "
        f"n={cfg.expansion_factor} tests={cfg.number_of_tests} | "
        f"{len(problems)} problems {'[MOCK]' if args.mock else ''}"
    )
    print(f"logging to {out_path}\n")

    for i, item in enumerate(problems, 1):
        print(f"[{i}/{len(problems)}] {item['name']}")
        for name in names:
            calls_before = llm.num_calls
            result = STRATEGY_FNS[name](item, llm, cfg, rng=rng, log=print)
            tally[name]["solved"] += int(result.solved)
            tally[name]["candidates"] += result.num_candidates
            tally[name]["calls"] += llm.num_calls - calls_before
            with open(out_path, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "name": result.name,
                            "strategy": result.strategy,
                            "solved": result.solved,
                            "solved_on_first_try": result.solved_on_first_try,
                            "iterations_used": result.iterations_used,
                            "num_candidates": result.num_candidates,
                            "final_code": result.final_code,
                        }
                    )
                    + "\n"
                )
        print()

    _print_comparison(tally, len(problems))
    return 0


def _print_comparison(tally: dict, total: int) -> None:
    header = ("strategy", "pass@1", "solved", "candidates", "llm_calls")
    print("=" * 64)
    print(
        f"{header[0]:<12}{header[1]:<10}{header[2]:<10}{header[3]:<12}{header[4]:<10}"
    )
    print("-" * 64)
    for name, t in tally.items():
        rate = t["solved"] / total if total else 0.0
        solved_str = f"{t['solved']}/{total}"
        print(
            f"{name:<12}{rate:<10.3f}{solved_str:<10}"
            f"{t['candidates']:<12}{t['calls']:<10}"
        )
    print("=" * 64)


if __name__ == "__main__":
    sys.exit(main())
