import math

from lats.node import Node, gather_context_from_tree


def test_unvisited_uct_is_inf():
    n = Node("x")
    assert n.uct() == float("inf")


def test_update_accumulates():
    n = Node("x")
    n.update(0.5)
    n.update(0.5)
    assert n.visits == 2
    assert n.value == 1.0


def test_backpropagate_updates_ancestors():
    root = Node("root")
    child = Node("child", parent=root, depth=1)
    grandchild = Node("gc", parent=child, depth=2)
    grandchild.backpropagate(1.0)
    assert grandchild.visits == child.visits == root.visits == 1
    assert root.value == child.value == grandchild.value == 1.0


def test_uct_formula():
    root = Node("root")
    root.visits = 4
    child = Node("c", parent=root)
    child.visits, child.value = 2, 1.0  # mean 0.5
    expected = 0.5 + 1.0 * math.sqrt(math.log(4) / 2)
    assert math.isclose(child.uct(1.0), expected)


def test_best_child_value_uses_mean_reward():
    root = Node("root")
    a = Node("a", parent=root)
    a.visits, a.value = 1, 0.9  # mean 0.9
    b = Node("b", parent=root)
    b.visits, b.value = 5, 2.0  # mean 0.4
    root.children = [a, b]
    assert root.best_child_value() is a


def test_gather_context_root_first_order():
    root = Node("root")
    root.test_feedback, root.reflection = "fb_root", "ref_root"
    child = Node("child", parent=root, depth=1)
    child.test_feedback, child.reflection = "fb_child", "ref_child"
    feedbacks, reflections = gather_context_from_tree(child)
    assert feedbacks == ["fb_root", "fb_child"]
    assert reflections == ["ref_root", "ref_child"]
