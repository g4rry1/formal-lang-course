from collections import Counter
from unittest.mock import Mock

import cfpq_data
import networkx as nx
import pydot
import pytest

from project.graph_utils import GraphInfo, create_two_cycles_graph, get_graph_info


def test_get_graph_info_from_csv(tmp_path, monkeypatch):
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(0, 1, label="a")
    graph.add_edge(0, 1, label="b")
    graph.add_edge(1, 2, label="a")
    graph.add_edge(2, 2, label="b")
    path = cfpq_data.graph_to_csv(graph, tmp_path / "graph.csv")
    download = Mock(return_value=path)
    monkeypatch.setattr(cfpq_data, "download", download)

    info = get_graph_info("example")

    assert isinstance(info, GraphInfo)
    assert info == (3, 5, {"a", "b"})
    assert info.vertex_count == 3
    assert info.edge_count == 5
    assert info.labels == {"a", "b"}
    download.assert_called_once_with("example")


@pytest.mark.parametrize("nodes", [[], [0], [0, 1, 2]])
def test_get_graph_info_without_edges(nodes, tmp_path, monkeypatch):
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(nodes)
    path = tmp_path / "graph.csv"
    monkeypatch.setattr(cfpq_data, "download", Mock(return_value=path))
    load_graph = Mock(return_value=graph)
    monkeypatch.setattr(cfpq_data, "graph_from_csv", load_graph)

    assert get_graph_info("example") == (len(nodes), 0, set())
    load_graph.assert_called_once_with(path)


@pytest.mark.parametrize(
    "n, m, labels",
    [(1, 1, ("a", "b")), (2, 3, ("first", "second")), (3, 2, ("a", "a"))],
)
@pytest.mark.parametrize("string_path", [False, True])
def test_create_two_cycles_graph(n, m, labels, string_path, tmp_path):
    path = tmp_path / "two cycles.dot"
    graph = create_two_cycles_graph(n, m, labels, str(path) if string_path else path)

    assert isinstance(graph, nx.MultiDiGraph)
    assert graph.number_of_nodes() == n + m + 1
    assert graph.number_of_edges() == n + m + 2
    cycles = list(nx.simple_cycles(graph))
    assert len(cycles) == 2
    assert sorted(map(len, cycles)) == sorted([n + 1, m + 1])
    assert set(cycles[0]) & set(cycles[1]) == {0}
    for label, nodes in [
        (labels[0], set(range(1, n + 1))),
        (labels[1], set(range(n + 1, n + m + 1))),
    ]:
        assert all(
            edge_label == label
            for u, v, edge_label in graph.edges(data="label")
            if u in nodes or v in nodes
        )

    (dot_graph,) = pydot.graph_from_dot_file(str(path))
    assert dot_graph.get_type() == "digraph"
    saved_graph = nx.drawing.nx_pydot.from_pydot(dot_graph)
    assert set(saved_graph.nodes) == {str(node) for node in graph.nodes}
    assert Counter(saved_graph.edges(data="label")) == Counter(
        (str(u), str(v), label) for u, v, label in graph.edges(data="label")
    )
