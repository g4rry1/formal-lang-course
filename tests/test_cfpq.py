"""Tests for weak Chomsky normal form and Hellings' algorithm."""

from itertools import product

import networkx as nx
from pyformlang.cfg import CFG, Terminal, Variable

from project.cfpq import cfg_to_weak_normal_form, hellings_based_cfpq


def test_weak_normal_form_preserves_nullable_language_and_rule_shapes():
    grammar = CFG.from_text(
        """
        S -> A | a S b | $
        A -> B
        B -> a b
        """
    )
    original_productions = set(grammar.productions)

    converted = cfg_to_weak_normal_form(grammar)

    assert grammar.productions == original_productions
    assert converted.start_symbol == grammar.start_symbol
    assert all(
        not production.body
        or (len(production.body) == 1 and isinstance(production.body[0], Terminal))
        or (
            len(production.body) == 2
            and all(isinstance(symbol, Variable) for symbol in production.body)
        )
        for production in converted.productions
    )
    for length in range(5):
        for word in product(("a", "b"), repeat=length):
            assert converted.contains(word) == grammar.contains(word)


def test_hellings_handles_empty_paths_and_nullable_recursive_grammar():
    graph = nx.DiGraph()
    graph.add_nodes_from((0, 1, 2, 3, 4, 9))
    graph.add_edges_from(
        [
            (0, 1, {"label": "a"}),
            (1, 2, {"label": "a"}),
            (2, 3, {"label": "b"}),
            (3, 4, {"label": "b"}),
        ]
    )
    grammar = CFG.from_text("S -> a S b | $")

    assert hellings_based_cfpq(grammar, graph) == {
        (vertex, vertex) for vertex in graph.nodes
    } | {(0, 4), (1, 3)}
    assert hellings_based_cfpq(grammar, graph, {0, 1, 100}, {3, 4, 100}) == {
        (0, 4),
        (1, 3),
    }


def test_hellings_handles_multiedges_and_nondefault_start_symbol():
    graph = nx.MultiDiGraph()
    graph.add_nodes_from((-2, 10, 42))
    graph.add_edge(-2, 10, label="a")
    graph.add_edge(-2, 10, label="c")
    graph.add_edge(10, 42, label="b")
    grammar = CFG.from_text(
        "Root -> A B\nA -> a\nB -> b", start_symbol=Variable("Root")
    )

    assert hellings_based_cfpq(grammar, graph, set(), set()) == {(-2, 42)}
    assert hellings_based_cfpq(grammar, graph, {10}, {42}) == set()
