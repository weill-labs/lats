"""End-to-end HotPotQA LATS with a deterministic MockLLM and a fake Wikipedia env."""

from lats.config import LATSConfig
from lats.hotpotqa.agent import Step, evaluate_state, generate_steps
from lats.hotpotqa.search import run_lats_hotpotqa
from lats.llm import MockLLM

QUESTION = {"id": "q1", "question": "What is the capital of France?", "answer": "Paris"}
FRANCE_PAGE = "France is a country in Europe. Paris is the capital of France."
CFG = LATSConfig(max_iters=5, expansion_factor=1, react_max_depth=5)


def _fake_env():
    return (
        lambda title: FRANCE_PAGE if title.lower() == "france" else None,  # get_page
        lambda q: ["France"],  # search_titles
    )


def test_search_solves_via_search_then_finish():
    def handler(messages, n):
        system, user = messages[0].content, messages[1].content
        if "[ROLE:value]" in system:
            return ["8"]
        if "[ROLE:reflect-qa]" in system:
            return ["search the country first"]
        # ReAct: once we've searched France, finish with Paris.
        if "Search[France]" in user:
            return ["Thought: The capital is Paris.\nAction: Finish[Paris]"] * n
        return ["Thought: I should look up France.\nAction: Search[France]"] * n

    get_page, search_titles = _fake_env()
    r = run_lats_hotpotqa(
        QUESTION, MockLLM(handler), CFG, get_page=get_page, search_titles=search_titles
    )
    assert r.solved and r.em
    assert r.predicted == "Paris"
    assert r.f1 == 1.0
    assert r.num_nodes >= 3  # root + search + finish


def test_search_unsolved_reflects_and_reports_wrong_answer():
    def handler(messages, n):
        system = messages[0].content
        if "[ROLE:value]" in system:
            return ["3"]
        if "[ROLE:reflect-qa]" in system:
            return ["wrong city; reconsider"]
        return ["Thought: It's London.\nAction: Finish[London]"] * n

    get_page, search_titles = _fake_env()
    r = run_lats_hotpotqa(
        QUESTION, MockLLM(handler), CFG, get_page=get_page, search_titles=search_titles
    )
    assert not r.solved and not r.em
    assert r.predicted == "London"


def test_generate_steps_parses_thought_and_action():
    llm = MockLLM(lambda m, n: ["Thought: reason here.\nAction: Search[X]"] * n)
    steps = generate_steps(llm, "q", [], [], n=2)
    assert steps == [("reason here.", "Search[X]"), ("reason here.", "Search[X]")]


def test_evaluate_state_normalizes_score():
    llm = MockLLM(lambda m, n: ["I'd rate this 7 out of 10"])
    assert evaluate_state(llm, "q", [Step("t", "Search[X]", "obs")]) == 0.7
