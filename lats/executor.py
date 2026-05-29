"""Run candidate code against tests in an isolated subprocess.

Two entry points:

- `execute(func, tests)`  -> ExecResult: run the LLM's *internal* (self-generated)
  asserts. The pass fraction becomes the MCTS reward; the formatted feedback is
  fed back into generation/reflection prompts.
- `evaluate(entry_point, func, hidden_test)` -> bool: run HumanEval's hidden
  `check(candidate)` to score the final answer.

Safety/design note: the official LATS executor `exec()`s model-generated code in
its own process via a thread with a timeout. We instead spawn a fresh
subprocess (`sys.executable`) and pass the untrusted code/tests as JSON on
stdin, so they are never string-interpolated into the harness. Each internal
test is bounded by a per-test SIGALRM inside the child, and the whole subprocess
has a hard wall-clock timeout as a backstop against hangs the signal can't break.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from typing import List

# ---------------------------------------------------------------------------
# Child harness. Fixed code — untrusted input arrives via stdin JSON only.
# It defines the function, runs each assert in a fresh namespace copy under a
# per-test alarm, and reports which passed/failed (with the offending call's
# output for failures, to make feedback actionable).
# ---------------------------------------------------------------------------
# SECURITY: this harness intentionally exec()s/eval()s LLM-generated code and
# tests. Running candidate code IS the task (it provides LATS's environment
# feedback / reward signal), so it cannot be replaced with ast.literal_eval.
# Untrusted strings arrive only via stdin JSON (never interpolated into this
# source), and every execution runs in a throwaway subprocess under SIGALRM +
# a wall-clock timeout. Do not run this on a key-bearing host without isolation.
_EXECUTE_HARNESS = r"""
import sys, json, ast, signal

def _timeout(signum, frame):
    raise TimeoutError()
signal.signal(signal.SIGALRM, _timeout)

data = json.load(sys.stdin)
func, tests, t = data["func"], data["tests"], int(data["timeout"]) or 1

ns = {}
def_ok, def_err = True, ""
try:
    exec("from typing import *\n" + func, ns)
except Exception as e:
    def_ok, def_err = False, f"{type(e).__name__}: {e}"

def call_output(test_src):
    # Evaluate the call on the left of the assert's comparison for a readable
    # "# output: ..." hint. Falls back to the whole test expression.
    try:
        parsed = ast.parse(test_src).body[0]
        node = getattr(parsed, "test", parsed)
        target = getattr(node, "left", node)
        expr = ast.unparse(target)
        signal.alarm(t)
        try:
            return repr(eval(expr, dict(ns)))
        finally:
            signal.alarm(0)
    except TimeoutError:
        return "TIMEOUT"
    except Exception as e:
        return f"{type(e).__name__}: {e}"

passed, failed = [], []
for test_src in tests:
    if not def_ok:
        failed.append({"test": test_src, "output": def_err})
        continue
    try:
        signal.alarm(t)
        exec(test_src, dict(ns))
        passed.append(test_src)
    except Exception:
        failed.append({"test": test_src, "output": call_output(test_src)})
    finally:
        signal.alarm(0)

print(json.dumps({"passed": passed, "failed": failed, "def_ok": def_ok}))
"""

_EVALUATE_HARNESS = r"""
import sys, json, signal

def _timeout(signum, frame):
    raise TimeoutError()
signal.signal(signal.SIGALRM, _timeout)

data = json.load(sys.stdin)
code = "from typing import *\n" + data["func"] + "\n\n" + data["test"] + \
       "\n\ncheck(" + data["entry_point"] + ")\n"
signal.alarm(int(data["timeout"]) or 1)
try:
    exec(code, {})
    print("PASS")
except Exception as e:
    print("FAIL")
"""


@dataclass
class ExecResult:
    is_passing: bool
    feedback: str
    num_passed: int
    num_tests: int

    @property
    def reward(self) -> float:
        """Fraction of internal tests passed — the MCTS value signal."""
        return self.num_passed / self.num_tests if self.num_tests else 0.0


def _run_child(
    harness: str, payload: dict, wall_timeout: float
) -> "subprocess.CompletedProcess | None":
    try:
        return subprocess.run(
            [sys.executable, "-c", harness],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=wall_timeout,
        )
    except subprocess.TimeoutExpired:
        return None


def _format_feedback(passed: List[str], failed: List[dict]) -> str:
    lines = ["Tests passed:"]
    lines += passed
    lines.append("")
    lines.append("Tests failed:")
    lines += [f"{f['test']} # output: {f['output']}" for f in failed]
    return "\n".join(lines)


def execute(func: str, tests: List[str], timeout: int = 5) -> ExecResult:
    """Run `tests` (assert strings) against `func`; summarise pass/fail."""
    if not tests:
        return ExecResult(
            is_passing=False, feedback="No tests.", num_passed=0, num_tests=0
        )

    wall = timeout * len(tests) + 5  # backstop if a per-test alarm can't fire
    proc = _run_child(
        _EXECUTE_HARNESS, {"func": func, "tests": tests, "timeout": timeout}, wall
    )

    if proc is None or proc.returncode != 0 or not proc.stdout.strip():
        # Whole subprocess hung or crashed: treat every test as failed.
        failed = [{"test": t, "output": "TIMEOUT or fatal error"} for t in tests]
        return ExecResult(False, _format_feedback([], failed), 0, len(tests))

    result = json.loads(proc.stdout.strip().splitlines()[-1])
    passed, failed = result["passed"], result["failed"]
    return ExecResult(
        is_passing=(len(failed) == 0),
        feedback=_format_feedback(passed, failed),
        num_passed=len(passed),
        num_tests=len(tests),
    )


def evaluate(entry_point: str, func: str, hidden_test: str, timeout: int = 10) -> bool:
    """Run HumanEval's hidden `check(candidate)` — the ground-truth scorer."""
    proc = _run_child(
        _EVALUATE_HARNESS,
        {
            "func": func,
            "test": hidden_test,
            "entry_point": entry_point,
            "timeout": timeout,
        },
        wall_timeout=timeout + 5,
    )
    return proc is not None and proc.returncode == 0 and proc.stdout.strip() == "PASS"
