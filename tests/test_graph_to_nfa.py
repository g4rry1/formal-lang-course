"""Behavioral tests for conversion of labeled graphs to NFAs."""

import cfpq_data
import networkx as nx
from pyformlang.finite_automaton import State

from project.graph_utils import create_two_cycles_graph, graph_to_nfa


def test_graph_to_nfa_respects_endpoints_and_parallel_edges():
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(range(5))
    graph.add_edges_from(
        [
            (0, 1, {"label": "a"}),
            (0, 1, {"label": "x"}),
            (0, 2, {"label": "a"}),
            (1, 3, {"label": "b"}),
            (2, 3, {"label": "c"}),
        ]
    )

    nfa = graph_to_nfa(graph, {0}, {3})

    assert not nfa.is_deterministic()
    assert State(4) in nfa.states  # An isolated vertex remains a state.
    assert nfa.accepts(["a", "b"])
    assert nfa.accepts(["x", "b"])
    assert nfa.accepts(["a", "c"])
    assert not nfa.accepts([])
    assert not nfa.accepts(["a"])
    assert not nfa.accepts(["b"])
    assert not nfa.accepts(["x", "c"])


def test_graph_to_nfa_uses_all_vertices_for_empty_state_sets():
    graph = nx.MultiDiGraph()
    graph.add_node(7)

    nfa = graph_to_nfa(graph, set(), set())

    assert nfa.accepts([])
    assert not nfa.accepts(["a"])


def test_graph_to_nfa_empty_graph_has_empty_language():
    nfa = graph_to_nfa(nx.MultiDiGraph(), set(), set())

    assert nfa.is_empty()


def test_graph_to_nfa_defaults_start_and_final_states_independently():
    graph = nx.MultiDiGraph()
    graph.add_nodes_from([0, 1, 2])
    graph.add_edge(0, 1, label="a")

    all_starts = graph_to_nfa(graph, set(), {2})
    assert all_starts.accepts([])
    assert not all_starts.accepts(["a"])

    all_finals = graph_to_nfa(graph, {0}, set())
    assert all_finals.accepts([])
    assert all_finals.accepts(["a"])


def test_graph_to_nfa_accepts_graph_loaded_from_csv(tmp_path):
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="edge")
    csv_path = cfpq_data.graph_to_csv(graph, tmp_path / "graph.csv")
    loaded_graph = cfpq_data.graph_from_csv(csv_path)

    nfa = graph_to_nfa(loaded_graph, {0}, {1})

    assert nfa.accepts(["edge"])
    assert not nfa.accepts([])


def test_graph_to_nfa_accepts_cycles_generated_by_task_one(tmp_path):
    graph = create_two_cycles_graph(1, 2, ("left", "right"), tmp_path / "cycles.dot")
    nfa = graph_to_nfa(graph, {0}, {0})

    assert nfa.accepts(["left", "left"])
    assert nfa.accepts(["right", "right", "right"])
    assert not nfa.accepts(["left"])
