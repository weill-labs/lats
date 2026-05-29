# Progress Log — LATS HumanEval reproduction

## Session 1 — 2026-05-29

### Phases 1–6 (on `master`, pushed to github.com/weill-labs/lats)
- Researched paper + official code (findings.md). Built clean-room LATS under `lats/`.
- Vendored dataset (160 problems). Secret from hgm/.env OAI_KEY → repo .env (gitignored).
- Tests 19/19; 20-problem gpt-3.5 run → **pass@1 = 17/20 = 0.850**.
- README + docs/DESIGN.md. Renamed repo+dir `lats-humaneval` → `lats`. Pushed public.

### Phase 7 — Baselines (branch `add-baselines`)
- Refactor: `result.py` (shared ProblemResult + `strategy`); `branch_history` → node.py.
- `strategies.py`: run_simple / run_reflexion (linear) / run_dfs (ToT, budgeted, no UCT/backprop).
- CLI `--strategy {lats,simple,reflexion,dfs,all}`; `all` → comparison table + per-strategy call attribution.
- `tests/test_strategies_mock.py`. Full suite **24/24 green**; ruff clean.

### Test / run results
- 4-strategy comparison, gpt-3.5-turbo, first 8 problems (logs/compare8.jsonl):

  | strategy | pass@1 | candidates | llm_calls |
  |---|---|---|---|
  | simple | 5/8 = 0.625 | 8 | 8 |
  | reflexion | 5/8 = 0.625 | 21 | 44 |
  | dfs | 5/8 = 0.625 | 41 | 51 |
  | lats | **6/8 = 0.750** | 40 | 75 |

- LATS uniquely solved `double_the_difference`; `fibfib` needed search (simple missed it).
- Honest caveat: N=8 is within sampling noise; `encrypt` solved only by `simple` (lucky sample).

### Next (optional)
- Larger comparison (`--strategy all --num-problems 50`) for a less noisy ranking.
- HotPotQA domain (adds real ReAct + Wikipedia env). Full GPT-4 sweep.
