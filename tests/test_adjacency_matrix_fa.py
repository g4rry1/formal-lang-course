"""Behavioral tests for sparse finite automata and their intersection."""

from itertools import product

from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, State, Symbol
from scipy.sparse import issparse

from project.adjacency_matrix_fa import AdjacencyMatrixFA
from project.rpq import intersect_automata


def test_matrix_nfa_matches_pyformlang_with_multiple_starts():
    nfa = NondeterministicFiniteAutomaton(states={State(-7), State(10), State(42)})
    nfa.add_start_state(-7)
    nfa.add_start_state(10)
    nfa.add_final_state(42)
    nfa.add_transition(-7, "a", 10)
    nfa.add_transition(-7, "a", 42)
    nfa.add_transition(10, "b", 42)
    nfa.add_transition(42, "a", 42)

    matrix_nfa = AdjacencyMatrixFA(nfa)

    assert matrix_nfa.adjacency_matrix.shape == (3, 3)
    assert all(issparse(matrix) for matrix in matrix_nfa.transition_matrices.values())
    for length in range(4):
        for word in product(("a", "b", "c"), repeat=length):
            assert matrix_nfa.accepts(iter(word)) == nfa.accepts(word)


def test_matrix_nfa_detects_empty_language_and_empty_word():
    disconnected = NondeterministicFiniteAutomaton(states={State(0), State(1)})
    disconnected.add_start_state(0)
    disconnected.add_final_state(1)
    disconnected.add_transition(1, "a", 1)
    matrix_nfa = AdjacencyMatrixFA(disconnected)

    assert matrix_nfa.is_empty()
    assert not matrix_nfa.accepts([])
    assert not matrix_nfa.accepts(["a"])

    disconnected.add_final_state(0)
    matrix_nfa = AdjacencyMatrixFA(disconnected)

    assert not matrix_nfa.is_empty()
    assert matrix_nfa.accepts([])
    assert not matrix_nfa.accepts(["a"])


def test_intersection_synchronizes_only_shared_labels():
    first = NondeterministicFiniteAutomaton()
    first.add_start_state(10)
    first.add_final_state(20)
    first.add_transition(10, "a", 20)
    first.add_transition(10, "b", 20)

    second = NondeterministicFiniteAutomaton()
    second.add_start_state(-2)
    second.add_final_state(-1)
    second.add_transition(-2, "b", -1)
    second.add_transition(-2, "c", -1)

    product_automaton = intersect_automata(
        AdjacencyMatrixFA(first), AdjacencyMatrixFA(second)
    )

    assert set(product_automaton.transition_matrices) == {Symbol("b")}
    assert product_automaton.accepts(["b"])
    assert not product_automaton.accepts(["a"])
    assert not product_automaton.accepts(["c"])
    assert not product_automaton.accepts([])
    assert not product_automaton.is_empty()


def test_intersection_of_disjoint_languages_is_empty():
    first = NondeterministicFiniteAutomaton()
    first.add_start_state(0)
    first.add_final_state(1)
    first.add_transition(0, "a", 1)

    second = NondeterministicFiniteAutomaton()
    second.add_start_state(0)
    second.add_final_state(1)
    second.add_transition(0, "b", 1)

    product_automaton = intersect_automata(
        AdjacencyMatrixFA(first), AdjacencyMatrixFA(second)
    )

    assert product_automaton.is_empty()
    assert not product_automaton.accepts(["a"])
    assert not product_automaton.accepts(["b"])
