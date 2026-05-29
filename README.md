# LATS — Language Agent Tree Search (clean-room reproduction)

A small, tested reimplementation of **Language Agent Tree Search (LATS)**
([arXiv:2310.04406](https://arxiv.org/abs/2310.04406), ICML 2024), built from the
paper's algorithm and the official repo's prompts, on the modern OpenAI SDK.

Two domains are implemented, sharing one domain-agnostic core (`node.py` UCT +
backprop, `llm.py`, `result.py`):
- **programming / HumanEval** — MCTS over code; reward = self-generated unit tests.
- **HotPotQA (reasoning + acting)** — ReAct (Thought/Action/Observation) over a
  Wikipedia search environment; reward = answer exact-match.

WebShop (decision-making) is the remaining paper domain and would slot in the
same way.

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
  config.py      LATSConfig + paper presets (gpt35 / gpt4) + react_max_depth
  llm.py         provider abstraction: OpenAIChat (openai>=1.0) + MockLLM
  node.py        MCTS Node, UCT, backprop, branch-context — shared by both domains
  result.py      ProblemResult (shared across strategies)
  # --- HumanEval domain ---
  prompts.py     the four prompt roles (simple / reflexion / reflect / tests)
  generator.py   LLM wrappers: internal_tests / simple_impl / reflexion_impls / self_reflection
  executor.py    subprocess-isolated test runner (execute) + hidden scorer (evaluate)
  parse.py       extract code from fenced / unfenced LLM output
  mcts.py        run_lats_on_problem — the search
  strategies.py  baselines: simple / reflexion / dfs
  dataset.py     load + subset HumanEval-Python
  run_lats.py    CLI (pass@1, --strategy {lats,simple,reflexion,dfs,all})
  data/humaneval-py.jsonl   vendored benchmark (MultiPL-E reworded, 160 problems)
  # --- HotPotQA domain ---
  hotpotqa/
    wikienv.py   ReAct Search/Lookup/Finish over Wikipedia (injectable HTTP)
    scoring.py   SQuAD-style normalize / exact_match / f1
    dataset.py   load official HotPotQA json/jsonl (+ vendored sample)
    prompts.py   ReAct / value-function / reflection prompts
    agent.py     Step, generate_steps, evaluate_state (LM value), reflect
    search.py    run_lats_hotpotqa — MCTS over ReAct trajectories
  run_hotpotqa.py  CLI (EM/F1 reporting)
tests/           44 tests, all API-free (MockLLM + injected env)
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

# Baselines and head-to-head comparison
uv run lats --strategy reflexion --num-problems 10   # one baseline
uv run lats --strategy all --num-problems 8          # simple/reflexion/dfs/lats + table

# Useful flags
--strategy {lats,simple,reflexion,dfs,all}
--max-iters N --expansion-factor N --number-of-tests N --temperature F
--exploration-weight F --start N --names NAME [NAME ...] --shuffle-seed N --output PATH
```

Results stream to stdout and a JSONL log (default `logs/lats_<ts>.jsonl`, one
row per problem+strategy with `solved`, `iterations_used`, `num_candidates`, `final_code`).

## HotPotQA domain (reasoning + acting)

ReAct over a Wikipedia environment, searched with MCTS. Each tree node is one
ReAct step (Thought + Action + Observation); the path to a node is a trajectory.
Expansion samples `n` next steps and *executes* each action in a per-branch
Wikipedia env; a terminal `Finish[answer]` gets the exact-match reward, a
non-terminal node gets the LM value-function estimate, and failed trajectories
add a reflection that feeds later expansions.

```bash
# Vendored sample (5 simple questions) — validates the pipeline
uv run lats-hotpotqa --num-problems 5 --model gpt-3.5-turbo
uv run lats-hotpotqa --mock --num-problems 2          # no API / no network

# Real multi-hop eval: point at the official HotPotQA dev file
uv run lats-hotpotqa --dataset hotpot_dev_distractor_v1.json --num-problems 50
```

### Results

**Pipeline sanity** (vendored single-fact sample, `gpt-3.5-turbo`): EM 5/5. This
only proves the full path works (real Wikipedia + ReAct + MCTS + value fn); the
questions rarely force deep Search→Lookup.

**Real multi-hop** — 20 questions from the official HotPotQA distractor dev split
(pulled via the HF datasets-server), `gpt-3.5-turbo`:

| config | EM | F1 |
|---|---|---|
| `max_iters=8`, verbose answers | 4/20 = 0.20 | 0.30 |
| `max_iters=12`, terse-answer prompt | **7/20 = 0.35** | **0.46** |

The jump came from a principled fix surfaced by the eval: the model reasoned yes/no
questions correctly but answered in sentences ("Yes, Scott Derrickson…" vs gold
"yes"), so the `Finish` prompt now requires short answer spans. **This is a
faithful partial reproduction with a known gap** vs the paper's ~0.6 GPT-3.5
number — the remaining misses are (a) hard 2-hop questions that exhaust the search
budget without a `Finish` (under-search) and (b) date/format normalization
near-misses. Closing the gap needs a larger search (bigger `n` / more iters) and
the paper's exact prompts. Reproduce with:

```bash
uv run lats-hotpotqa --dataset <hotpot_dev.json> --num-problems 20 --max-iters 12
```

## Tests

```bash
uv run pytest -q     # 44 tests, no API key needed (uses MockLLM + injected env)
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

## Baselines & comparison

`--strategy all` runs four methods on the same problems with the same generator,
executor, and prompts — so the only thing that varies is the **search**:

| Strategy | Search | Branching | Backtracks? | Value signal |
|---|---|---|---|---|
| `simple` | none (1 sample) | — | no | — |
| `reflexion` | linear chain | no | no | internal tests (to trigger reflect) |
| `dfs` | depth-first (ToT) | yes (`n`) | yes, to best sibling | internal-test pass rate |
| `lats` | MCTS (UCT) | yes (`n`) | yes, via UCT + backprop | internal-test pass rate |

Head-to-head, `gpt-3.5-turbo`, **first 8 problems** (`max_iters=4, n=3, tests=4`):

| Strategy | pass@1 | candidates generated | LLM calls |
|---|---|---|---|
| `simple` | 5/8 = 0.625 | 8 | 8 |
| `reflexion` | 5/8 = 0.625 | 21 | 44 |
| `dfs` | 5/8 = 0.625 | 41 | 51 |
| **`lats`** | **6/8 = 0.750** | 40 | 75 |

LATS came out ahead, uniquely solving `double_the_difference`. All four solved
the four easy problems from the first sample; `fibfib` required search (the three
search methods got it, `simple` did not). Cost rises with search depth/branching:
`simple` (8 calls) < `reflexion` (44) < `dfs` (51) < `lats` (75).

**Honest caveat:** 8 problems is far too few to separate these methods
statistically — the ranking is within sampling noise. `encrypt` is a telling
artifact: `simple`'s single lucky sample passed while the search methods' (independent,
temperature-0.8) first samples differed and search didn't recover within budget.
The paper's clean separation shows up at full benchmark scale, not on 8 problems.
Reproduce/extend with `uv run lats --strategy all --num-problems 50`.

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
