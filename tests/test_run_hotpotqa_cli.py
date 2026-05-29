"""Smoke test the HotPotQA CLI end-to-end in --mock mode (no API, no network)."""

from lats.run_hotpotqa import main


def test_cli_mock_runs(tmp_path, capsys):
    out = tmp_path / "out.jsonl"
    rc = main(
        [
            "--mock",
            "--num-problems",
            "2",
            "--max-iters",
            "3",
            "--expansion-factor",
            "1",
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    captured = capsys.readouterr().out
    assert "EM =" in captured
    # one jsonl row per question
    assert len(out.read_text().strip().splitlines()) == 2
