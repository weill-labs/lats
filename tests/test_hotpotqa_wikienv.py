from lats.hotpotqa.wikienv import WikipediaEnv, parse_action

PAGE = (
    "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.\n"
    "It was designed by Gustave Eiffel. The tower is 330 metres tall."
)


def _env():
    pages = {"Eiffel Tower": PAGE}
    return WikipediaEnv(
        get_page=lambda t: pages.get(t),
        search_titles=lambda q: ["Eiffel Tower", "Eiffel (disambiguation)"],
    )


def test_parse_action():
    assert parse_action("Search[Eiffel Tower]") == ("Search", "Eiffel Tower")
    assert parse_action("Finish[42]") == ("Finish", "42")
    assert parse_action("not an action") is None


def test_search_found_returns_first_paragraph():
    env = _env()
    obs = env.act("Search", "Eiffel Tower")
    assert obs.startswith("The Eiffel Tower is a wrought-iron")


def test_search_not_found_returns_similar():
    env = _env()
    obs = env.act("Search", "Eifel Towr")
    assert "Could not find" in obs and "Eiffel Tower" in obs


def test_lookup_iterates_matching_sentences():
    env = _env()
    env.act("Search", "Eiffel Tower")
    first = env.act("Lookup", "Eiffel")
    assert first.startswith("(Result 1/")
    # second matching sentence ("designed by Gustave Eiffel")
    second = env.act("Lookup", "Eiffel")
    assert "Gustave Eiffel" in second
    # exhausted
    assert "No more results" in env.act("Lookup", "Eiffel")


def test_lookup_before_search():
    assert "Search" in _env().act("Lookup", "x")


def test_finish_sets_answer_and_done():
    env = _env()
    obs = env.act("Finish", "Paris")
    assert env.done and env.answer == "Paris"
    assert "Paris" in obs


def test_invalid_action():
    assert "Invalid action" in _env().act("Jump", "x")
