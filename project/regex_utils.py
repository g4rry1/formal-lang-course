"""Convert regular expressions into finite automata."""

from pyformlang.finite_automaton import DeterministicFiniteAutomaton
from pyformlang.regular_expression import Regex


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    """Convert a pyformlang regular expression to a minimal DFA.

    Args:
        regex: Expression in the syntax understood by ``Regex``.

    Returns:
        A minimal DFA accepting the same language.
    """

    parsed_regex = Regex(regex)
    epsilon_nfa = parsed_regex.to_epsilon_nfa()
    dfa = epsilon_nfa.to_deterministic()
    minimal_dfa = dfa.minimize()
    return minimal_dfa
