import os
import torch
import networkx as nx
import json

class GraphTorch:
    """
    Graph representation that stores adjacency matrix as a PyTorch tensor
    for GPU-based operations like edge contractions.
    """
    def __init__(self, nx_graph: nx.Graph, device=None):
        """
        Initialize from a NetworkX graph
        :param nx_graph: input graph (NetworkX)
        :param device: 'cpu' or 'cuda', default auto-detect
        """
        self.nx_graph = nx_graph
        self.nodes = list(nx_graph.nodes())
        self.index_map = {node: i for i, node in enumerate(self.nodes)}  
        self.reverse_map = {i: node for node, i in self.index_map.items()}  

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.adj = self.to_adjacency_tensor().to(self.device)

    def to_adjacency_tensor(self):
        """Convert NetworkX graph to PyTorch adjacency matrix."""
        n = len(self.nodes)
        adj = torch.zeros((n, n), dtype=torch.int8)
        for u, v in self.nx_graph.edges():
            i, j = self.index_map[u], self.index_map[v]
            adj[i, j] = 1
            adj[j, i] = 1
        return adj

    def num_nodes(self):
        return self.adj.shape[0]

    def get_edges(self):
        """Return list of edges by their labels."""
        edges = []
        idx_edges = torch.nonzero(self.adj, as_tuple=False)
        for i, j in idx_edges:
            if i < j:
                edges.append((self.reverse_map[int(i)], self.reverse_map[int(j)]))
        return edges

    def to_networkx(self):
        """Convert the current adjacency matrix back to a NetworkX graph."""
        G = nx.Graph()
        G.add_nodes_from(self.nodes)
        for i in range(self.num_nodes()):
            for j in range(i + 1, self.num_nodes()):
                if self.adj[i, j] == 1:
                    G.add_edge(self.reverse_map[i], self.reverse_map[j])
        return G

    def save_to_json(self, filename, output_dir=None):
        """
        Save adjacency matrix and graph structure to JSON file.
        Args:
            filename: Name of the file to save
            output_dir: Output Directory
        """
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        graphs_dir = os.path.join(project_root, "graphs")

        if output_dir:
            save_dir = os.path.join(graphs_dir, output_dir)
        else:
            save_dir = graphs_dir

        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, filename)
        
        data = {
            "nodes": self.nodes,
            "index_map": self.index_map,
            "device": self.device,
            "adjacency_matrix": self.adj.cpu().tolist(),
            "graph_type": "undirected"
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)
        print(f"[+] Graph saved to {filepath}")

        return filepath

    def __repr__(self):
        return f"GraphTorch(num_nodes={self.num_nodes()}, device='{self.device}')"

    def fan_graph(n):
        """Creates Fan Graph"""
        G = nx.Graph()
        for i in range(n):
            G.add_node(i)
        for i in range(1, n - 1):
            G.add_edge(i, i + 1)
        for i in range(1, n):
            G.add_edge(0, i)
        return G