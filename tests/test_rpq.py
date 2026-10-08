"""Tests for regular path queries on labeled multigraphs."""

from collections import deque
from random import Random

import networkx as nx
import pytest
from pyformlang.finite_automaton import Symbol

from project.regex_utils import regex_to_dfa
from project.rpq import tensor_based_rpq


def _reference_rpq(regex, graph, start_nodes, final_nodes):
    """Use ordinary graph search as an independent query oracle."""

    automaton = regex_to_dfa(regex)
    transitions = automaton.to_dict()
    result = set()
    for source in start_nodes:
        queue = deque((state, source) for state in automaton.start_states)
        visited = set(queue)
        while queue:
            state, vertex = queue.popleft()
            if state in automaton.final_states and vertex in final_nodes:
                result.add((source, vertex))
            for _, target, label in graph.out_edges(vertex, data="label"):
                next_state = transitions.get(state, {}).get(Symbol(label))
                if next_state is not None and (next_state, target) not in visited:
                    visited.add((next_state, target))
                    queue.append((next_state, target))
    return result


@pytest.mark.parametrize("regex", ["a b", "a*", "(a|b)* b", "epsilon"])
def test_rpq_matches_graph_search_with_noncontiguous_vertices(regex):
    graph = nx.MultiDiGraph()
    graph.add_nodes_from((-7, 10, 42, 99))
    graph.add_edges_from(
        [
            (-7, 10, {"label": "a"}),
            (-7, 10, {"label": "b"}),
            (10, 42, {"label": "b"}),
            (42, 42, {"label": "a"}),
            (42, 99, {"label": "b"}),
        ]
    )
    start_nodes = {-7, 42, 99}
    final_nodes = {-7, 42, 99}

    assert tensor_based_rpq(regex, graph, start_nodes, final_nodes) == (
        _reference_rpq(regex, graph, start_nodes, final_nodes)
    )


def test_rpq_uses_all_vertices_for_empty_endpoint_sets():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")

    assert tensor_based_rpq("a", graph, set(), {1}) == {(0, 1)}
    assert tensor_based_rpq("a", graph, {0}, set()) == {(0, 1)}
    assert tensor_based_rpq("a*", graph, set(), set()) == {
        (0, 0),
        (0, 1),
        (1, 1),
    }


def test_rpq_matches_search_on_small_random_graphs():
    random = Random(73)
    regexes = ("a b", "(a|b)*", "a* b", "b a*")
    for _ in range(6):
        graph = nx.MultiDiGraph()
        graph.add_nodes_from(range(4))
        for _ in range(8):
            graph.add_edge(
                random.randrange(4),
                random.randrange(4),
                label=random.choice(("a", "b", "c")),
            )
        start_nodes = {0, 2}
        final_nodes = {1, 3}
        for regex in regexes:
            assert tensor_based_rpq(regex, graph, start_nodes, final_nodes) == (
                _reference_rpq(regex, graph, start_nodes, final_nodes)
            )


def test_rpq_ignores_vertices_not_in_the_graph():
    graph = nx.MultiDiGraph()
    graph.add_node(0)

    assert tensor_based_rpq("a*", graph, {7}, {7}) == set()
    assert tensor_based_rpq("a*", graph, {0, 7}, {0, 7}) == {(0, 0)}
