from lats.hotpotqa.scoring import exact_match, f1_score, normalize_answer


def test_normalize_strips_articles_punct_case():
    assert normalize_answer("The White House.") == "white house"
    assert normalize_answer("  A   Cat! ") == "cat"


def test_exact_match_is_normalized():
    assert exact_match("Paris", "paris")
    assert exact_match("the   Beatles", "Beatles")
    assert not exact_match("Paris", "London")


def test_f1_partial_overlap():
    # pred "Barack Obama" vs gold "Obama": 1 common tok / pred 2, recall 1/1 -> F1 = 2*.5*1/1.5
    assert abs(f1_score("Barack Obama", "Obama") - (2 * 0.5 * 1.0 / 1.5)) < 1e-9


def test_f1_perfect_and_zero():
    assert f1_score("New York City", "new york city") == 1.0
    assert f1_score("cat", "dog") == 0.0
