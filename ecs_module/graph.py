import os
import json
import networkx as nx
from typing import Any, Dict, List, Tuple, Optional


class Graph:
    """
    Lightweight graph wrapper around a NetworkX graph.
    Stores:
      - the underlying NetworkX graph
      - a fixed node ordering
      - an adjacency matrix as a plain Python list-of-lists (no PyTorch)
    """
    def __init__(self, nx_graph: nx.Graph, device: Optional[str] = None):
        """
        Initialize from a NetworkX graph.
        """
        self.nx_graph: nx.Graph = nx_graph.copy()

        self.nodes: List[Any] = list(self.nx_graph.nodes())
        self._node_index: Dict[Any, int] = {n: i for i, n in enumerate(self.nodes)}

        self.adj: List[List[int]] = self._build_adjacency_matrix()

        self.device = device or "cpu"

    def _build_adjacency_matrix(self) -> List[List[int]]:
        """Create a 0/1 adjacency matrix."""
        n = len(self.nodes)
        mat = [[0] * n for _ in range(n)]

        for u, v in self.nx_graph.edges():
            i = self._node_index[u]
            j = self._node_index[v]
            mat[i][j] = 1
            mat[j][i] = 1

        return mat

    def to_networkx(self) -> nx.Graph:
        return self.nx_graph.copy()

    def num_nodes(self) -> int:
        return len(self.nodes)

    def get_edges(self) -> List[Tuple[Any, Any]]:
        return list(self.nx_graph.edges())

    def save_to_json(self, filename: str, output_dir: Optional[str] = None):
        """Save nodes + adjacency matrix."""
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        graphs_dir = os.path.join(project_root, "graphs")

        if output_dir:
            out_dir = os.path.join(graphs_dir, output_dir)
        else:
            out_dir = graphs_dir

        os.makedirs(out_dir, exist_ok=True)
        filepath = os.path.join(out_dir, filename)

        data = {
            "nodes": self.nodes,
            "adjacency_matrix": self.adj,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)

        return filepath

    def __repr__(self):
        return f"GraphTorch(num_nodes={self.num_nodes()}, device='{self.device}')"

    @staticmethod
    def fan_graph(n: int) -> nx.Graph:
        G = nx.Graph()
        for i in range(n):
            G.add_node(i)

        for i in range(1, n - 1):
            G.add_edge(i, i + 1)

        for i in range(1, n):
            G.add_edge(0, i)

        return G
