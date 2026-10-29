"""Weak Chomsky normal form and context-free path query algorithms."""

from collections import defaultdict, deque

import networkx as nx
from pyformlang.cfg import CFG, Production, Terminal, Variable
from scipy.sparse import coo_array, csr_array

from project.graph_utils import graph_to_nfa


def cfg_to_weak_normal_form(cfg: CFG) -> CFG:
    """Convert a CFG to binary, terminal, and (if needed) epsilon rules.

    ``CFG.to_normal_form`` removes the empty word from the language. Restore
    it for the original start symbol when the input grammar generates it.
    """

    normal_form = cfg.to_normal_form()
    if not cfg.generate_epsilon():
        return normal_form

    return CFG(
        start_symbol=normal_form.start_symbol,
        productions={
            *normal_form.productions,
            Production(normal_form.start_symbol, []),
        },
    )


def hellings_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    """Find vertex pairs connected by a path whose labels belong to ``cfg``.

    Missing or empty endpoint sets select every graph vertex, as in the
    regular path query functions. Intermediate vertices are unrestricted.
    """

    vertices = set(graph.nodes)
    starts = vertices if not start_nodes else vertices & set(start_nodes)
    finals = vertices if not final_nodes else vertices & set(final_nodes)
    if not starts or not finals:
        return set()

    normal_form = cfg_to_weak_normal_form(cfg)
    terminal_heads: dict[Terminal, set[Variable]] = defaultdict(set)
    binary_by_left: dict[Variable, set[tuple[Variable, Variable]]] = defaultdict(set)
    binary_by_right: dict[Variable, set[tuple[Variable, Variable]]] = defaultdict(set)
    epsilon_heads: set[Variable] = set()

    for production in normal_form.productions:
        body = production.body
        if not body:
            epsilon_heads.add(production.head)
        elif len(body) == 1:
            terminal_heads[body[0]].add(production.head)
        else:
            left, right = body
            binary_by_left[left].add((right, production.head))
            binary_by_right[right].add((left, production.head))

    pending: deque[tuple[Variable, int, int]] = deque()
    discovered: set[tuple[Variable, int, int]] = set()

    def add_fact(variable: Variable, source: int, target: int) -> None:
        fact = (variable, source, target)
        if fact not in discovered:
            discovered.add(fact)
            pending.append(fact)

    for variable in epsilon_heads:
        for vertex in vertices:
            add_fact(variable, vertex, vertex)

    graph_automaton = graph_to_nfa(graph, set(), set())
    for source, transitions in graph_automaton.to_dict().items():
        for label, targets in transitions.items():
            for variable in terminal_heads.get(Terminal(label.value), ()):
                for target in targets:
                    add_fact(variable, source.value, target.value)

    # For each variable, index known paths by their first and last vertex.
    paths_from: dict[tuple[Variable, int], set[int]] = defaultdict(set)
    paths_to: dict[tuple[Variable, int], set[int]] = defaultdict(set)

    while pending:
        variable, source, target = pending.popleft()
        paths_from[(variable, source)].add(target)
        paths_to[(variable, target)].add(source)

        for right, head in binary_by_left.get(variable, ()):
            for end in paths_from.get((right, target), ()):
                add_fact(head, source, end)

        for left, head in binary_by_right.get(variable, ()):
            for beginning in paths_to.get((left, source), ()):
                add_fact(head, beginning, target)

    return {
        (source, target)
        for variable, source, target in discovered
        if variable == normal_form.start_symbol
        and source in starts
        and target in finals
    }


def matrix_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    """Find vertex pairs accepted by ``cfg`` using sparse Boolean matrices.

    Each nonterminal has a matrix of currently known paths. Terminal and
    epsilon rules seed the matrices; binary rules add matrix products until
    no new paths can be derived. Empty endpoint sets select all vertices.
    """

    vertices = tuple(graph.nodes)
    vertex_set = set(vertices)
    starts = vertex_set if not start_nodes else vertex_set & set(start_nodes)
    finals = vertex_set if not final_nodes else vertex_set & set(final_nodes)
    if not starts or not finals:
        return set()

    normal_form = cfg_to_weak_normal_form(cfg)
    vertex_to_index = {vertex: index for index, vertex in enumerate(vertices)}
    size = len(vertices)
    terminal_heads: dict[Terminal, set[Variable]] = defaultdict(set)
    binary_rules: list[tuple[Variable, Variable, Variable]] = []
    rows: dict[Variable, list[int]] = defaultdict(list)
    columns: dict[Variable, list[int]] = defaultdict(list)

    for production in normal_form.productions:
        body = production.body
        if not body:
            rows[production.head].extend(range(size))
            columns[production.head].extend(range(size))
        elif len(body) == 1:
            terminal_heads[body[0]].add(production.head)
        else:
            left, right = body
            binary_rules.append((production.head, left, right))

    for source, target, label in graph.edges(data="label"):
        for head in terminal_heads.get(Terminal(label), ()):
            rows[head].append(vertex_to_index[source])
            columns[head].append(vertex_to_index[target])

    matrices: dict[Variable, csr_array] = {}
    for variable in normal_form.variables | {normal_form.start_symbol}:
        indices = rows[variable], columns[variable]
        matrices[variable] = coo_array(
            ([True] * len(indices[0]), indices), shape=(size, size), dtype=bool
        ).tocsr()

    while True:
        changed = False
        for head, left, right in binary_rules:
            product = matrices[left] @ matrices[right]
            if product.nnz:
                updated = (matrices[head] + product).tocsr()
                if updated.nnz > matrices[head].nnz:
                    matrices[head] = updated
                    changed = True
        if not changed:
            break

    answer = matrices[normal_form.start_symbol].tocoo()
    return {
        (vertices[source], vertices[target])
        for source, target in zip(answer.row, answer.col)
        if vertices[source] in starts and vertices[target] in finals
    }
