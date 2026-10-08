"""Behavioral tests for regular-expression conversion."""

import pytest

from project.regex_utils import regex_to_dfa


@pytest.mark.parametrize(
    "expression, accepted_words, rejected_words",
    [
        (
            "a b*",
            [("a",), ("a", "b"), ("a", "b", "b")],
            [(), ("b",), ("a", "a")],
        ),
        (
            "a|b",
            [("a",), ("b",)],
            [(), ("a", "b")],
        ),
        (
            "ab",
            [("ab",)],
            [(), ("a", "b")],
        ),
        (
            "epsilon|a",
            [(), ("a",)],
            [("b",), ("a", "a")],
        ),
    ],
)
def test_regex_to_dfa_accepts_and_rejects_sample_words(
    expression, accepted_words, rejected_words
):
    dfa = regex_to_dfa(expression)

    for word in accepted_words:
        assert dfa.accepts(word), (expression, word)
    for word in rejected_words:
        assert not dfa.accepts(word), (expression, word)


def test_regex_to_dfa_merges_equivalent_states():
    dfa = regex_to_dfa("a* | a")

    assert dfa.is_deterministic()
    assert len(dfa.states) == 1
    assert dfa.accepts([])
    assert dfa.accepts(["a", "a"])
