"""Integration tests for regex and graph automata."""

import networkx as nx

from project.graph_utils import graph_to_nfa
from project.regex_utils import regex_to_dfa


def test_large_graph_and_regex_define_same_language():
    """Compare a 121-vertex graph with a 120-choice regular expression."""

    step_count = 120
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(range(step_count + 1))
    for vertex in range(step_count):
        graph.add_edge(vertex, vertex + 1, label="a")
        graph.add_edge(vertex, vertex + 1, label="b")

    expression = " ".join(["(a|b)"] * step_count)
    nfa = graph_to_nfa(graph, {0}, {step_count})
    dfa = regex_to_dfa(expression)

    assert dfa.is_equivalent_to(nfa)

    alternating_word = ["a" if index % 2 == 0 else "b" for index in range(step_count)]
    invalid_label_word = alternating_word.copy()
    invalid_label_word[step_count // 2] = "c"

    for accepted_word in (alternating_word, ["b"] * step_count):
        assert nfa.accepts(accepted_word)
        assert dfa.accepts(accepted_word)

    for rejected_word in (
        alternating_word[:-1],
        alternating_word + ["a"],
        invalid_label_word,
    ):
        assert not nfa.accepts(rejected_word)
        assert not dfa.accepts(rejected_word)
