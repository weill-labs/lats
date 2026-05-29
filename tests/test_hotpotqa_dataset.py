import json

from lats.hotpotqa.dataset import load_hotpotqa, select_subset


def test_load_vendored_sample():
    data = load_hotpotqa()
    assert len(data) >= 5
    assert {"id", "question", "answer"} <= set(data[0])
    assert data[0]["answer"] == "Paris"


def test_load_official_array_format(tmp_path):
    f = tmp_path / "dev.json"
    f.write_text(
        json.dumps(
            [{"_id": "x1", "question": "Q?", "answer": "A", "type": "comparison"}]
        )
    )
    data = load_hotpotqa(f)
    assert data == [{"id": "x1", "question": "Q?", "answer": "A"}]


def test_load_jsonl_format(tmp_path):
    f = tmp_path / "d.jsonl"
    f.write_text(
        '{"id": "a", "question": "Q1", "answer": "A1"}\n{"id": "b", "question": "Q2", "answer": "A2"}\n'
    )
    assert [d["id"] for d in load_hotpotqa(f)] == ["a", "b"]


def test_select_subset_by_count_and_ids():
    data = load_hotpotqa()
    assert len(select_subset(data, num_problems=2)) == 2
    assert [d["id"] for d in select_subset(data, ids=["sample-3"])] == ["sample-3"]
