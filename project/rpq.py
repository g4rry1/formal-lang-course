"""Regular path queries using sparse automata intersections."""

import networkx as nx
from pyformlang.finite_automaton import Symbol
from scipy.sparse import csr_array, eye_array, kron

from project.adjacency_matrix_fa import AdjacencyMatrixFA
from project.graph_utils import graph_to_nfa
from project.regex_utils import regex_to_dfa


def intersect_automata(
    automaton1: AdjacencyMatrixFA, automaton2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    """Intersect two automata by synchronizing equally labeled transitions."""

    states = tuple(
        (left, right) for left in automaton1.states for right in automaton2.states
    )
    start_states = {
        (left, right)
        for left in automaton1.start_states
        for right in automaton2.start_states
    }
    final_states = {
        (left, right)
        for left in automaton1.final_states
        for right in automaton2.final_states
    }
    shared_symbols = (
        automaton1.transition_matrices.keys() & automaton2.transition_matrices.keys()
    )
    transition_matrices: dict[Symbol, csr_array] = {
        symbol: kron(
            automaton1.transition_matrices[symbol],
            automaton2.transition_matrices[symbol],
            format="csr",
        )
        for symbol in shared_symbols
    }
    return AdjacencyMatrixFA._from_matrices(
        states, start_states, final_states, transition_matrices
    )


def _transitive_closure(adjacency: csr_array) -> csr_array:
    """Include empty paths and repeatedly double the maximum path length."""

    reachable = (
        eye_array(adjacency.shape[0], format="csr", dtype=bool) + adjacency
    ).tocsr()
    while True:
        extended = (reachable @ reachable).tocsr()
        extended.eliminate_zeros()
        if extended.nnz == reachable.nnz:
            return reachable
        reachable = extended


def tensor_based_rpq(
    regex: str,
    graph: nx.MultiDiGraph,
    start_nodes: set[int],
    final_nodes: set[int],
) -> set[tuple[int, int]]:
    """Find requested vertex pairs connected by a path matching ``regex``."""

    graph_nodes = set(graph.nodes)
    selected_starts = (set(start_nodes) if start_nodes else graph_nodes) & graph_nodes
    selected_finals = (set(final_nodes) if final_nodes else graph_nodes) & graph_nodes
    if not selected_starts or not selected_finals:
        return set()

    regex_automaton = AdjacencyMatrixFA(regex_to_dfa(regex))
    graph_automaton = AdjacencyMatrixFA(
        graph_to_nfa(graph, selected_starts, selected_finals)
    )
    product = intersect_automata(regex_automaton, graph_automaton)
    if not product.start_states or not product.final_states:
        return set()

    reachable = _transitive_closure(product.adjacency_matrix)
    final_indices = {product.state_to_index[state] for state in product.final_states}
    answer = set()
    for start_state in product.start_states:
        row = reachable[[product.state_to_index[start_state]], :]
        for target_index in final_indices.intersection(row.indices):
            answer.add((start_state[1].value, product.states[target_index][1].value))
    return answer
