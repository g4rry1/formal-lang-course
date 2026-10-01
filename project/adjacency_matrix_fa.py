"""Finite automata represented by sparse, label-indexed adjacency matrices."""

from collections.abc import Iterable

from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, Symbol
from scipy.sparse import coo_array, csr_array


class AdjacencyMatrixFA:
    """An NFA with one Boolean adjacency matrix for each transition label.

    States may have arbitrary hashable values. Matrix indices are internal and
    do not depend on the values used to name states in the original automaton.
    """

    def __init__(self, automaton: NondeterministicFiniteAutomaton) -> None:
        self.states = tuple(automaton.states)
        self.state_to_index = {state: index for index, state in enumerate(self.states)}
        self.start_states = set(automaton.start_states)
        self.final_states = set(automaton.final_states)

        edges: dict[Symbol, tuple[list[int], list[int]]] = {}
        for source, transitions in automaton.to_dict().items():
            for symbol, targets in transitions.items():
                rows, columns = edges.setdefault(symbol, ([], []))
                # A DFA stores one destination, whereas an NFA stores a set.
                for target in (
                    targets if isinstance(targets, (set, frozenset)) else (targets,)
                ):
                    rows.append(self.state_to_index[source])
                    columns.append(self.state_to_index[target])

        size = len(self.states)
        self.transition_matrices = {
            symbol: coo_array(
                ([True] * len(rows), (rows, columns)),
                shape=(size, size),
                dtype=bool,
            ).tocsr()
            for symbol, (rows, columns) in edges.items()
        }
        self.adjacency_matrix = self._combined_adjacency()

    @classmethod
    def _from_matrices(
        cls,
        states: tuple,
        start_states: set,
        final_states: set,
        transition_matrices: dict[Symbol, csr_array],
    ) -> "AdjacencyMatrixFA":
        """Build an automaton whose transitions are already sparse matrices."""

        result = cls.__new__(cls)
        result.states = states
        result.state_to_index = {state: index for index, state in enumerate(states)}
        result.start_states = start_states
        result.final_states = final_states
        result.transition_matrices = transition_matrices
        result.adjacency_matrix = result._combined_adjacency()
        return result

    def _combined_adjacency(self) -> csr_array:
        size = len(self.states)
        adjacency = csr_array((size, size), dtype=bool)
        for matrix in self.transition_matrices.values():
            adjacency += matrix
        return adjacency

    def accepts(self, word: Iterable[Symbol]) -> bool:
        """Check whether a path from an initial to a final state reads ``word``."""

        if not self.start_states or not self.final_states:
            return False

        start_indices = [self.state_to_index[state] for state in self.start_states]
        active = csr_array(
            ([True] * len(start_indices), ([0] * len(start_indices), start_indices)),
            shape=(1, len(self.states)),
            dtype=bool,
        )
        for symbol in word:
            matrix = self.transition_matrices.get(Symbol(symbol))
            if matrix is None:
                return False
            active = (active @ matrix).tocsr()
            if active.nnz == 0:
                return False

        return any(active[0, self.state_to_index[state]] for state in self.final_states)

    def is_empty(self) -> bool:
        """Check whether any final state is reachable from an initial state."""

        if not self.start_states or not self.final_states:
            return True

        start_indices = [self.state_to_index[state] for state in self.start_states]
        reachable = csr_array(
            ([True] * len(start_indices), ([0] * len(start_indices), start_indices)),
            shape=(1, len(self.states)),
            dtype=bool,
        )
        final_indices = {self.state_to_index[state] for state in self.final_states}
        while True:
            if final_indices.intersection(reachable.indices):
                return False
            extended = (reachable + reachable @ self.adjacency_matrix).tocsr()
            if extended.nnz == reachable.nnz:
                return True
            reachable = extended
