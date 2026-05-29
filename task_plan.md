# Task Plan — Reproduce LATS on HumanEval (clean-room)

## Goal
Build a clean, documented, tested reimplementation of **LATS (arXiv:2310.04406)** for the
**programming / HumanEval** domain, faithful to the paper's algorithm and prompts, on the
modern OpenAI SDK. Prove it works end-to-end on a small subset (~5–20 problems) and report
pass@1.

## Scope decisions (confirmed with user)
- Domain: **Programming / HumanEval** (self-contained; reward = self-generated internal tests).
- Model: **OpenAI GPT** (default `gpt-3.5-turbo` for cheap subset runs; `gpt-4` selectable).
- Style: **Clean-room reimplementation** (our code; reuse paper's prompts + hyperparameters).
- Runs: **small validation subset**, not the full 164-problem benchmark.

## Architecture (target files)
```
lats-humaneval/
  pyproject.toml            # uv-managed, deps: openai, tenacity, (dev) pytest
  README.md                 # paper summary, algorithm, usage, results
  .gitignore                # .venv, __pycache__, logs/, .env
  .env.example              # OPENAI_API_KEY
  docs/DESIGN.md            # architecture + decision log
  lats/
    __init__.py
    config.py               # dataclass: model, max_iters, n, number_of_tests, temp, c, timeout
    llm.py                  # provider abstraction; OpenAIChat + MockLLM; retries
    prompts.py              # all system prompts + few-shots (from paper)
    generator.py            # internal_tests / func_impl_simple / func_impl_reflexion / self_reflection
    executor.py             # subprocess-isolated execute() + evaluate(); feedback formatting
    parse.py                # parse_code_block / add_code_block
    node.py                 # Node + UCT + gather_context_from_tree
    mcts.py                 # run_lats_on_problem (clean select/expand/simulate/backprop)
    dataset.py              # load humaneval-py.jsonl, subset selection
    run_lats.py             # CLI entrypoint; loops problems; computes pass@1; writes logs
    data/humaneval-py.jsonl # vendored 164-problem dataset
  tests/
    test_parse.py
    test_executor.py
    test_node.py
    test_mcts_mock.py       # end-to-end with MockLLM (no API), asserts trivial problem solved
```

## Phases

### Phase 1 — Repo scaffold & deps — `complete`
- [x] Create repo dir + git init + subdirs.
- [x] Planning files (this).
- [x] `pyproject.toml` (uv), `.gitignore`, `.env.example`.
- [x] Vendor `data/humaneval-py.jsonl` (160 problems; MultiPL-E reworded set).
- [x] `uv sync` → openai 2.38.0, tenacity, dotenv, pytest. `.env` written from hgm OAI_KEY (validated).

### Phase 2 — Core primitives (no API) — `complete`
- [x] `parse.py` (+ test). `executor.py` subprocess-isolated (+ test on known good/bad funcs).
- [x] `node.py` UCT/backprop (+ test). `config.py`.
- Gate: 16/16 passed (`pytest tests/test_parse.py tests/test_executor.py tests/test_node.py`).

### Phase 3 — LLM layer + generator — `complete`
- [x] `llm.py`: OpenAIChat (openai>=1.0) + MockLLM; tenacity retries; `n`/temperature; call counters.
- [x] `prompts.py`: paper prompts/few-shots. `generator.py`: 4 generation fns.

### Phase 4 — MCTS loop + runner — `complete`
- [x] `mcts.py`: clean select→expand(n)→simulate→backprop; reward=internal_frac+real.
- [x] `run_lats.py`: CLI (model/max-iters/expansion-factor/number-of-tests/num-problems/mock/output).
- [x] `dataset.py`.

### Phase 5 — End-to-end validation — `complete`
- [x] `test_mcts_mock.py`: MockLLM solves trivial problem (fast path + reflexion path). 19/19 pytest green.
- [x] Mock run `uv run lats --mock` over 3 problems (zero API cost) — runs clean.
- [x] 2-problem real API run: 2/2 solved (validated live path).
- [x] 20-problem real run (gpt-3.5-turbo): **pass@1 = 17/20 = 0.850** (13 first-try + 4 by search).

### Phase 6 — Docs & wrap-up — `complete`
- [x] README.md + docs/DESIGN.md decision log.
- [x] Fill README results table from run20.jsonl.
- [x] Final commit + pushed: github.com/weill-labs/lats (public). Local dir renamed `lats-humaneval` → `lats`.

### Phase 7 — Baseline strategies (feature branch `add-baselines`) — `in_progress`
- [x] Refactor: shared `result.py` (ProblemResult + `strategy` field); `branch_history` → node.py.
- [x] `strategies.py`: run_simple (pass@1 base), run_reflexion (linear), run_dfs (ToT; budgeted; no UCT/backprop).
- [x] `run_lats.py`: `--strategy {lats,simple,reflexion,dfs,all}`; `all` prints comparison + per-strategy call attribution.
- [x] `tests/test_strategies_mock.py`. Full suite 24/24 green; ruff clean.
- [ ] Real `--strategy all` on 8 problems — RUNNING (logs/compare8.jsonl); fill comparison table.
- [ ] README baselines section + DESIGN D9. Commit on branch, push, open PR → master.

## Resolved decisions
1. Repo named **`lats`** (org: weill-labs, public), HumanEval = first domain.
2. `OPENAI_API_KEY`: sourced from `~/github/weill-labs/hgm/.env` (`OAI_KEY`) → repo `.env` (gitignored).
3. Subset for the headline run: 20 problems, gpt-3.5-turbo.

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| (none yet) | | |
