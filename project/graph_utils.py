"""Load, create, and convert labeled graphs."""

from pathlib import Path
from typing import NamedTuple

import cfpq_data
import networkx as nx
from networkx.drawing.nx_pydot import write_dot
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton


class GraphInfo(NamedTuple):
    """Vertex and edge counts together with distinct edge labels."""

    vertex_count: int
    edge_count: int
    labels: set[str]


def get_graph_info(name: str) -> GraphInfo:
    """Get statistics for a named graph from the CFPQ dataset.

    Args:
        name: Graph name accepted by ``cfpq_data.download``.

    Returns:
        Vertex count, edge count, and distinct edge labels.
    """

    graph = cfpq_data.graph_from_csv(cfpq_data.download(name))
    return GraphInfo(
        graph.number_of_nodes(),
        graph.number_of_edges(),
        {label for _, _, label in graph.edges(data="label")},
    )


def create_two_cycles_graph(
    n: int,
    m: int,
    labels: tuple[str, str],
    path: str | Path,
) -> nx.MultiDiGraph:
    """Create two labeled cycles sharing one vertex and save them as DOT.

    The node counts exclude the vertex shared by both cycles.

    Args:
        n: Number of first-cycle vertices excluding the shared vertex.
        m: Number of second-cycle vertices excluding the shared vertex.
        labels: Edge labels for the first and second cycles, in order.
        path: Destination DOT file path.

    Returns:
        The generated directed multigraph.
    """

    graph = cfpq_data.labeled_two_cycles_graph(n, m, labels=labels)
    write_dot(graph, path)
    return graph


def graph_to_nfa(
    graph: nx.MultiDiGraph, start_states: set[int], final_states: set[int]
) -> NondeterministicFiniteAutomaton:
    """Build an NFA from the vertices and labeled edges of a graph.

    Args:
        graph: Directed multigraph with a ``label`` attribute on each edge.
        start_states: Initial vertices; an empty set selects all vertices.
        final_states: Accepting vertices; an empty set selects all vertices.

    Returns:
        An NFA whose transitions follow the labeled graph edges.
    """

    nfa = NondeterministicFiniteAutomaton(states=set(graph.nodes))

    for state in start_states or graph.nodes:
        nfa.add_start_state(state)
    for state in final_states or graph.nodes:
        nfa.add_final_state(state)

    for source, target, label in graph.edges(data="label"):
        nfa.add_transition(source, label, target)

    return nfa
