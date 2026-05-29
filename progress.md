# Progress Log — LATS HumanEval reproduction

## Session 1 — 2026-05-29

### Done (all phases complete)
- Researched paper (arXiv:2310.04406) + official `programming/` code; captured to findings.md.
- Built clean-room implementation under `lats/`:
  config, llm (OpenAIChat + MockLLM), prompts, generator, executor (subprocess-isolated),
  parse, node (UCT/backprop), mcts (clean select/expand/simulate/backprop), dataset, run_lats CLI.
- Vendored dataset (160 problems). `uv sync` (openai 2.38.0, tenacity, dotenv, pytest).
- Secret: copied hgm/.env OAI_KEY → repo .env (gitignored), validated against OpenAI API.
- Wrote README.md + docs/DESIGN.md (decision log).

### Test results
- `uv run pytest -q` → 19/19 passed (parse, executor, node, end-to-end mock). No API key needed.
- `uv run lats --mock --num-problems 3` → runs full loop, 0 API cost, pass@1 0/3 (stub; plumbing only).
- 2-problem real run (gpt-3.5-turbo) → 2/2 solved.
- 20-problem real run (gpt-3.5-turbo, preset iters=4/n=3/tests=4) → **pass@1 = 17/20 = 0.850**
  (13 first-try, 4 by search; 3 misses exhausted iters cleanly). 136 LLM calls, 176 completions.
  Logs: logs/run20.jsonl, logs/run20.out.

### Key observations
- Fast path dominates on easy problems (root solution passes hidden tests, 0 iterations).
- MCTS search + reflection demonstrably rescues failures (encrypt 2it, fibfib 1it, do_algebra last-iter).
- Subprocess executor timeouts hold under real generated code (no hangs on the 3 hard misses).

### Next (optional, not started)
- Full 164-problem sweep / GPT-4 preset for the headline number.
- DFS/Reflexion/ReAct baselines; token-cost accounting; pass@k.
