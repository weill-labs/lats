# Design & Decision Log — LATS / HumanEval reproduction

## Goal
Clean-room reimplementation of LATS (arXiv:2310.04406) for HumanEval: faithful to
the paper's algorithm and prompts, modern OpenAI SDK, tested without an API key,
validated on a small subset.

## Architecture

```
run_lats (CLI)
  └─ dataset.load_dataset/select_subset
  └─ llm.OpenAIChat | MockLLM          (provider abstraction)
  └─ mcts.run_lats_on_problem          (one problem = one search tree)
       ├─ generator.internal_tests     ┐
       ├─ generator.simple_impl        │  prompts.py → llm.chat → parse.py
       ├─ generator.reflexion_impls    │
       ├─ generator.self_reflection    ┘
       ├─ executor.execute             (internal tests → reward + feedback)
       ├─ executor.evaluate            (hidden check → final score)
       └─ node.Node / UCT / backprop
```

Dependency direction is one-way (CLI → search → generator/executor → llm/parse/
node), so each layer is unit-testable in isolation.

## The programming-variant reward (the crux)
The paper frames LATS with an LM value function, but for code the official repo's
reward is **execution-based**: `reward_internal = (#self-generated tests passed) /
(#tests)`, plus `reward_real = 1` if the candidate passes the *hidden* tests.
We keep exactly this. The "value function" is therefore the self-test pass rate,
which is why the domain is self-contained (no external environment).

## Decision Log

### D1 — Clean-room rewrite, reuse prompts (not a port)
User chose "clean-room". The prompts *are* the method, so we reuse the official
instructions/few-shots (trimmed) while writing our own structure and control flow.

### D2 — Modern OpenAI SDK + provider abstraction
Official targets `openai<1.0` (`openai.ChatCompletion.create`), which is removed.
We use `openai>=1.0` (`client.chat.completions.create`). `LLM.chat()` returns a
list of `n` completions so one expansion is one round-trip via OpenAI's native
`n`. `MockLLM` implements the same contract for tests.

### D3 — Rewrote the MCTS loop
The official `programming/mcts.py` expansion does, inside `for _ in range(n)`:
`for child in node.children: ... break`, re-simulating earlier children and
backpropagating a loop-dangling `child`. We implement the intended cycle:
select (UCT to leaf) → expand `n` children → simulate each → hidden-check when
internal-passing or last iteration → backprop each. Reward math is unchanged.

### D4 — Unvisited UCT = +inf (was: raw value)
Standard MCTS forces exploration of unvisited nodes. The official returned the
raw value for `visits==0`. We return `+inf`. Low impact in practice (children are
simulated immediately on creation, so they have visits≥1 before the next
selection), but it's the correct rule.

### D5 — Subprocess-isolated executor
Running model-generated code is the task (it's the environment feedback), so
`exec`/`eval` is intentional and cannot be replaced with `ast.literal_eval`.
For safety we run it in a throwaway subprocess (`sys.executable -c`), pass the
untrusted code/tests as **stdin JSON** (never interpolated into the harness),
bound each internal test with `SIGALRM`, and add a wall-clock timeout backstop.
This is stronger isolation than the official thread-in-driver approach and can't
leak runaway threads. Caveat: still not a security sandbox — don't run untrusted
output on a key-bearing host without container/VM isolation.

### D6 — Re-generate internal tests each iteration
Matches the official behaviour. Adds diversity to the reward signal so search
doesn't overfit a single (possibly wrong) test set. We use a seeded RNG for the
"sample N of the generated asserts" step so runs are reproducible.

### D7 — Dataset vendored
`lats/data/humaneval-py.jsonl` is the official MultiPL-E "reworded"
HumanEval-Python set (160 problems; a few of the 164 are dropped in that
translation). Vendoring keeps runs offline and reproducible.

### D8 — Secret handling
`OPENAI_API_KEY` lives only in gitignored `.env` (loaded via python-dotenv from
the repo root regardless of cwd). It was copied from `hgm/.env`'s `OAI_KEY`
without ever appearing in argv, stdout, or shell history.

### D9 — Baseline strategies share LATS's machinery
`strategies.py` adds `simple`, `reflexion`, and `dfs`, all reusing the same
`generator`, `executor`, prompts, and `Node`, so a comparison isolates the
*search algorithm* rather than confounding it with prompt/eval differences.
- **simple** — one zero-shot sample scored on hidden tests (pass@1 base).
- **reflexion** — a single chain (no branching): run internal tests → reflect →
  regenerate with accumulated reflection memory, up to `max_iters` times.
- **dfs** — Tree-of-Thoughts-style depth-first search: expand `n`, recurse into
  the highest-internal-reward child first, backtrack on dead ends. No UCT and no
  backpropagation (the distinction from LATS). Bounded by a `max_iters × n`
  candidate budget so generation cost is comparable to LATS.

`ProblemResult` moved to `result.py` (with a `strategy` field) so `mcts.py` and
`strategies.py` share one type without importing each other. The CLI gained
`--strategy {lats,simple,reflexion,dfs,all}`; `all` prints a comparison table and
attributes LLM calls per strategy by snapshotting the counter.

Note on ReAct: pure ReAct (thought→action→observation against an external
environment) is a baseline for the *acting* domains (HotPotQA/WebShop), not
HumanEval. For code its act-observe loop reduces to Reflexion's test-feedback
loop, which is what we implement. ReAct proper will land with the HotPotQA domain.

## Known limitations
- Small-subset validation only; not the full 164-problem benchmark or GPT-4.
- `gpt-4` preset (`max_iters=8, n=5`) is wired but untested here (cost).
- Cost/latency scale with `max_iters × n` LLM calls per problem.
- DFS worst-case fan-out is capped by a candidate budget; that cap is logged
  implicitly via `num_candidates` but DFS may stop before exhausting the tree.

## Possible extensions
- DFS/Reflexion/ReAct baselines for comparison (official repo has them).
- Token-usage accounting for $ cost per problem.
- pass@k (k>1) and the full benchmark sweep.
