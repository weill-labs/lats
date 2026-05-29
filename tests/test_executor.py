from lats.executor import evaluate, execute

GOOD = "def add(a, b):\n    return a + b"
BUGGY = "def add(a, b):\n    return a - b"
TESTS = ["assert add(1, 2) == 3", "assert add(0, 0) == 0", "assert add(-1, 1) == 0"]


def test_all_pass():
    r = execute(GOOD, TESTS)
    assert r.is_passing
    assert r.num_passed == r.num_tests == 3
    assert r.reward == 1.0


def test_partial_fail_reports_output():
    r = execute(BUGGY, TESTS)
    assert not r.is_passing
    # add(1,2) -> -1 fails; add(0,0) -> 0 passes; add(-1,1) -> -2 fails
    assert r.num_passed == 1
    assert r.reward == 1 / 3
    assert "# output:" in r.feedback


def test_infinite_loop_does_not_hang():
    loop = "def add(a, b):\n    while True:\n        pass"
    r = execute(loop, ["assert add(1, 2) == 3"], timeout=1)
    assert not r.is_passing
    assert r.num_passed == 0


def test_syntax_error_func_all_fail():
    r = execute("def add(a, b)\n    return a + b", TESTS)  # missing colon
    assert not r.is_passing
    assert r.num_passed == 0


def test_evaluate_hidden_check_pass_and_fail():
    hidden = "def check(candidate):\n    assert candidate(2, 3) == 5\n    assert candidate(10, 5) == 15"
    assert evaluate("add", GOOD, hidden) is True
    assert evaluate("add", BUGGY, hidden) is False
