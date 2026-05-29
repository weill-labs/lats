# LATS — Language Agent Tree Search (clean-room reproduction)

A small, tested reimplementation of **Language Agent Tree Search (LATS)**
([arXiv:2310.04406](https://arxiv.org/abs/2310.04406), ICML 2024), built from the
paper's algorithm and the official repo's prompts, on the modern OpenAI SDK.

The **first implemented domain is programming / HumanEval**. The code is layered
(a provider-agnostic LLM interface, a pluggable executor and dataset loader) so
the other LATS domains — HotPotQA (reasoning+acting) and WebShop
(decision-making) — can be added alongside it later.

> LATS wraps an LLM in a Monte Carlo Tree Search loop. The *same* model is the
> agent (writes code), the value-function helper (its self-generated tests score
> each candidate), and the reflector (explains failures so the next attempt
> improves). No fine-tuning — it's gradient-free.

## How it works (programming variant)

HumanEval's real unit tests are hidden, so LATS **generates its own tests** and
uses the fraction passed as the search reward. The hidden tests are only touched
to score the final answer.

```
internal tests  ─ LLM writes asserts (the value signal)
root            ─ an initial "simple" solution
                  └─ if it already passes the hidden tests → done (pass@1 fast path)
repeat max_iters:
  selection     ─ descend from root by UCT to a leaf
  expansion     ─ sample n children, each conditioned on the branch's
                  accumulated (code, test-feedback, reflection) history
  simulation    ─ run each child on the internal tests → reward = #passed / #tests
  hidden check  ─ a child passing internal tests (or on the last iter) is scored
                  on the hidden tests; a real pass ends the search
  backprop      ─ add reward to the child and every ancestor
```

`UCT(child) = value/visits + c·√(ln(parent.visits)/visits)` — the standard
exploit-vs-explore rule (same as AlphaGo), over LLM-generated children.

## Layout

```
lats/
  config.py      LATSConfig + paper presets (gpt35 / gpt4)
  llm.py         provider abstraction: OpenAIChat (openai>=1.0) + MockLLM
  prompts.py     the four prompt roles (simple / reflexion / reflect / tests)
  generator.py   LLM wrappers: internal_tests / simple_impl / reflexion_impls / self_reflection
  executor.py    subprocess-isolated test runner (execute) + hidden scorer (evaluate)
  parse.py       extract code from fenced / unfenced LLM output
  node.py        MCTS Node, UCT, backprop, branch-context gathering
  mcts.py        run_lats_on_problem — the search
  dataset.py     load + subset HumanEval-Python
  run_lats.py    CLI entrypoint (pass@1 reporting)
  data/humaneval-py.jsonl   vendored benchmark (MultiPL-E reworded, 160 problems)
tests/           parse / executor / node / end-to-end mock (no API)
docs/DESIGN.md   architecture + decision log
```

## Setup

```bash
uv sync                      # creates .venv, installs deps (openai, tenacity, dotenv, pytest)
cp .env.example .env         # then put your key in it:  export OPENAI_API_KEY=sk-...
```

## Usage

```bash
# Full loop, no API (plumbing smoke test)
uv run lats --mock --num-problems 3

# Real run on a subset
uv run lats --num-problems 20 --model gpt-3.5-turbo
uv run lats --num-problems 10 --model gpt-4          # uses the gpt4 preset

# Useful flags
--max-iters N --expansion-factor N --number-of-tests N --temperature F
--exploration-weight F --start N --names NAME [NAME ...] --shuffle-seed N --output PATH
```

Results stream to stdout and a JSONL log (default `logs/lats_<ts>.jsonl`, one
row per problem with `solved`, `iterations_used`, `num_candidates`, `final_code`).

## Tests

```bash
uv run pytest -q     # 19 tests, no API key needed (uses MockLLM)
```

## Results

Subset runs with `gpt-3.5-turbo` (paper preset: `max_iters=4, n=3, tests=4`):

| Subset | pass@1 | breakdown |
|---|---|---|
| first 2 (validation) | 2/2 = 1.000 | both solved on first try |
| first 20 | **17/20 = 0.850** | 13 zero-shot + 4 rescued by MCTS search; 136 LLM calls total |

In the 20-problem run, 13 problems were solved by the initial solution (no search
needed) and 4 of the remaining 7 were solved by the tree search + reflection loop
(e.g. `encrypt` in 2 iterations, `fibfib` in 1). The 3 misses each exhausted all
4 iterations and terminated cleanly. This ~85% tracks the paper's reported
GPT-3.5 HumanEval result.

These are small-subset sanity numbers, not the paper's full 164-problem figure
(92.7% GPT-4). Reproducing the headline number needs the full set + GPT-4:

```bash
uv run lats --num-problems 200 --model gpt-4    # full set (cost: many GPT-4 calls)
```

## Differences from the official implementation

This is a clean-room rewrite, not a port. Same algorithm and prompts; cleaner,
tested code. Notable choices (see `docs/DESIGN.md` for the full log):

1. **Modern OpenAI SDK.** The official code targets the removed
   `openai.ChatCompletion.create`; we use the `openai>=1.0` client.
2. **Correct MCTS loop.** The official expansion loop re-iterates over
   `node.children` with an early `break` and backpropagates a dangling variable;
   we implement a textbook select → expand(n) → simulate → backprop, keeping the
   reward semantics (`internal_fraction + real_pass`).
3. **Subprocess-isolated execution.** Candidate code runs in a throwaway
   subprocess with per-test `SIGALRM` + a wall-clock backstop, instead of
   `exec`-on-a-thread in the driver process.
4. **`MockLLM`** makes the entire search testable without an API key.

## Credits

Original paper & code: Andy Zhou, Kai Yan, Michal Shlapentokh-Rothman, Haohan
Wang, Yu-Xiong Wang — <https://github.com/lapisrocks/LanguageAgentTreeSearch>.
