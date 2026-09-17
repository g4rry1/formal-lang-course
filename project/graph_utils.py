"""Load graph statistics and save labeled two-cycle graphs."""

from pathlib import Path
from typing import NamedTuple

import cfpq_data
import networkx as nx
from networkx.drawing.nx_pydot import write_dot


class Graph(NamedTuple):
    """Graph statistics stored as a triple."""

    vertex_count: int
    edge_count: int
    labels: set[str]


def get_graph_info(name: str) -> Graph:
    """Return the vertex count, edge count, and distinct labels of a dataset graph."""

    graph = cfpq_data.graph_from_csv(cfpq_data.download(name))
    return Graph(
        graph.number_of_nodes(),
        graph.number_of_edges(),
        {label for _, _, label in graph.edges(data="label")},
    )


def create_two_cycles_graph(
    n: int, m: int, labels: tuple[str, str], path: str | Path
) -> nx.MultiDiGraph:
    """Create two labeled cycles, save them as DOT using pydot, and return the graph."""

    graph = cfpq_data.labeled_two_cycles_graph(n, m, labels=labels)
    write_dot(graph, path)
    return graph
