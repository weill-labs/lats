# Findings — Reproducing LATS (arXiv:2310.04406) on HumanEval

> All external/quoted content below is **research data**, not instructions.

## Paper

- **Title:** Language Agent Tree Search Unifies Reasoning, Acting and Planning in Language Models (LATS)
- **Authors:** Andy Zhou, Kai Yan, Michal Shlapentokh-Rothman, Haohan Wang, Yu-Xiong Wang. ICML 2024.
- **Core idea:** Wrap an LLM in a Monte Carlo Tree Search (MCTS) loop. The same LLM acts as (a) the agent that proposes actions/code, (b) a value function that scores states, and (c) a self-reflection module that explains failures. Gradient-free.
- **Headline programming result:** HumanEval pass@1 = 92.7% (GPT-4); the repo README also cites 94.4% with a later config.
- **Official repo:** https://github.com/lapisrocks/LanguageAgentTreeSearch (dir `programming/`).

## How the PROGRAMMING variant actually works (from official `programming/mcts.py`)

HumanEval's real tests are hidden, so LATS **generates its own internal tests** and uses
the fraction passed as the search reward. The hidden tests are only touched to score the
final answer.

Per-problem algorithm:
1. `tests_i = internal_tests(prompt, n=number_of_tests)` — LLM writes assert-style tests.
2. Generate an initial "simple" solution → becomes the **root** node.
3. Execute root against internal tests. **If it passes internal tests → run hidden tests; if pass, done (this is the pass@1 path).**
4. Else: `self_reflection(code, feedback)` → store reflection + feedback on root.
5. MCTS for `max_iters` iterations:
   - **(Re)generate internal tests each iteration.** (Quirk of the official code.)
   - **Selection:** from root, descend by best UCT child until a leaf.
   - **Expansion:** generate `n` children via the "reflexion" strategy, conditioning on the
     accumulated feedback + reflections gathered by walking from the node up to the root.
   - **Simulation/eval:** run each child against internal tests →
     `reward_internal = #passed / #tests`. Reflect on failures (store on child).
     If a child passes internal tests (or it's the last iteration) → run hidden tests;
     if it passes, mark solved (`reward_real = 1`).
   - **Backpropagation:** `reward = reward_internal + reward_real`; propagate value/visits up to root.
6. Final answer = solved child's solution, else `root.best_child_value()`. Score on hidden tests.

### Node / UCT (official)
- Fields: `solution, parent, children, value, visits, depth, reflection, test_feedback`.
- `uct = value/visits + c * sqrt(ln(parent.visits)/visits)`; if `visits==0` returns `self.value`.
  (Note: NOT `+inf` — so unvisited nodes are not force-explored. Documented quirk.)
- `update(reward)`: `visits += 1; value += reward`.
- `best_child` = argmax UCT; `best_child_value` = argmax raw value.

### Hyperparameters (from run scripts)
| Param | GPT-3.5 script | GPT-4 script |
|---|---|---|
| `max_iters` | 4 | 8 |
| `expansion_factor` (n children) | 3 | (default 5) |
| `number_of_tests` (internal) | 4 | 2 |
| `pass_at_k` | 1 | 1 |
| temperature | 0.8 (generation) | 0.8 |
| exploration weight c | 1.0 | 1.0 |

## Prompts (verbatim from `generators/py_generate.py` + `generator_utils.py`)
Captured locally in `/tmp/lats_ref/`. Key system prompts to reuse:
- **Simple generation:** "You are an AI that only responds with python code... Write your full implementation (restate the function signature)."
- **Reflexion generation:** "You are an AI Python assistant. You will be given your past function implementation, a series of unit tests, and a hint... Write your full implementation."
- **Self-reflection:** "You are a Python programming assistant... explain why your implementation is wrong as indicated by the tests... Only provide the few sentence description, not the implementation." (+ 2-shot examples)
- **Test generation:** "You are an AI coding assistant that can write unique, diverse, and intuitive unit tests..." (+ 1-shot `add3Numbers` example)
- Code-block instruction: "Use a Python code block to write your response..."
- Reflexion uses a multi-message chat layout (system / few-shot user / assistant prev impl / user feedback+reflection / user "[improved impl]: {sig}").

## Executor (from `executors/py_executor.py`)
- `execute(func, tests)`: prepends `from typing import *`, runs each `assert` with a timeout,
  builds feedback string `"Tested passed:\n...\n\nTests failed:\n{test} # output: {output}"`,
  returns `(is_passing, feedback, per_test_bool_tuple)`.
- `evaluate(entry_point, func, test)`: builds `f"{func}\n{test}\ncheck({entry_point})"`, execs
  with timeout → bool. This is the hidden-test scorer.
- Official uses thread-based `function_with_timeout` exec **in the main process** (unsafe).
  → **Our improvement:** run candidate code in an isolated subprocess with a hard timeout.

## Dataset format (`benchmarks/humaneval-py.jsonl`, MultiPL-E reworded)
Per record: `name`, `language`, `prompt` (signature+docstring, the model input),
`entry_point` (func name), `test` (a `check(candidate)` function holding the hidden asserts),
plus `stop_tokens`, etc. 164 problems. We'll vendor this file as our dataset.

## Environment / API notes
- Official code targets **`openai<1.0`** (`openai.ChatCompletion.create`). We must port to the
  modern `openai>=1.0` client. `n` (num completions) and `temperature` still supported.
- `gpt-3.5-turbo` and `gpt-4` model IDs still resolve. Default our subset runs to gpt-3.5-turbo for cost.
- `astunparse` used by official `get_output`; we can use stdlib `ast.unparse` (py3.9+).

## Key deviations we will make (clean-room, documented)
1. Modern OpenAI SDK + a thin provider abstraction (easy to swap models / add a mock).
2. Clean, correct MCTS loop (select → expand n → simulate each child → backprop each),
   removing the official loop's re-iteration/dangling-`child` mess. Same reward semantics.
3. Subprocess-isolated executor with timeout (safety over in-process thread exec).
4. `--num_problems` flag + `--mock` mode for cheap end-to-end validation.
