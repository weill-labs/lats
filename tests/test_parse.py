from lats.parse import add_code_block, parse_code_block


def test_language_fence():
    text = "Here you go:\n```python\ndef f(x):\n    return x + 1\n```\nDone."
    assert parse_code_block(text) == "def f(x):\n    return x + 1"


def test_bare_fence():
    text = "```\ndef f():\n    return 1\n```"
    assert parse_code_block(text) == "def f():\n    return 1"


def test_def_scan_fallback_no_fence():
    text = "Sure!\ndef f(x):\n    return x * 2\n"
    assert "def f(x):" in parse_code_block(text)
    assert "return x * 2" in parse_code_block(text)


def test_returns_none_when_no_code():
    assert parse_code_block("just prose, no code here") is None


def test_add_code_block_roundtrips():
    wrapped = add_code_block("def f(): return 1")
    assert wrapped.startswith("```python\n") and wrapped.endswith("\n```")
    assert parse_code_block(wrapped) == "def f(): return 1"
