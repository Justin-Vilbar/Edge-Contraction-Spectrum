import networkx as nx
import json
import os
from .graph import Graph


class EdgeContraction:
    """
    Handles batching of edge contractions and stores unique contraction levels.
    Tracks parent → child contraction relationships between levels.
    """
    def __init__(self, json_path: str = "", output_dir: str | None = None):
        self.graph = self.load_from_json(json_path)

        self.contraction_levels: dict[int, list[Graph]] = {0: [self.graph]}

        self.level_graph_ids: dict[int, list[int]] = {0: [0]}
        self.next_global_id: int = 1

        self.edges: list[tuple[int, int]] = []

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        graphs_dir = os.path.join(project_root, "graphs")

        if output_dir:
            self.output_path = os.path.join(graphs_dir, output_dir)
        else:
            self.output_path = graphs_dir

        os.makedirs(self.output_path, exist_ok=True)

    def load_from_json(self, json_path: str) -> Graph:
        """Load a graph from adjacency JSON file."""
        if not json_path:
            raise ValueError("JSON graph path not provided.")

        with open(json_path, "r") as f:
            data = json.load(f)

        nodes = data["nodes"]
        adj_matrix = data["adjacency_matrix"]

        G = nx.Graph()
        G.add_nodes_from(nodes)

        n = len(nodes)
        for i in range(n):
            for j in range(i + 1, n):
                if adj_matrix[i][j] == 1:
                    G.add_edge(nodes[i], nodes[j])

        return Graph(G)

    def contract_edge(self, graph_torch: Graph, u, v) -> Graph:
        """Contract edge (u, v) into one vertex and return the resulting Graph."""
        G = graph_torch.to_networkx()
        G = nx.contracted_nodes(G, u, v, self_loops=False)
        return Graph(G)

    def _find_isomorphic_index(self, new_graph: Graph, graphs: list[Graph]) -> int | None:
        """Return index of an isomorphic graph or None if not found."""
        G_new = new_graph.to_networkx()

        for idx, g in enumerate(graphs):
            if nx.is_isomorphic(G_new, g.to_networkx()):
                return idx
        return None

    def generate_next_level(self, level: int):
        next_level_graphs: list[Graph] = []
        parent_child_pairs: list[tuple[int, int]] = []

        current_graphs = self.contraction_levels[level]

        for parent_idx, graph in enumerate(current_graphs):
            edges = graph.get_edges()
            for (u, v) in edges:
                contracted_graph = self.contract_edge(graph, u, v)

                existing_idx = self._find_isomorphic_index(contracted_graph, next_level_graphs)
                if existing_idx is None:
                    child_idx = len(next_level_graphs)
                    next_level_graphs.append(contracted_graph)
                else:
                    child_idx = existing_idx

                parent_child_pairs.append((parent_idx, child_idx))

        return next_level_graphs, parent_child_pairs

    def build_spectrum(self, max_levels: int | None = None):
        level = 0
        print("Building edge contraction spectrum...")

        while True:
            if max_levels is not None and level >= max_levels:
                print(f"Stopping at max_levels = {max_levels}")
                break

            next_level, parent_child_pairs = self.generate_next_level(level)

            if not next_level:
                print("No more contractions possible. Stopping.")
                break

            next_level_idx = level + 1
            self.contraction_levels[next_level_idx] = next_level
            print(f"Level {next_level_idx}: {len(next_level)} non-isomorphic graphs")

            child_global_ids: list[int] = []
            for _ in next_level:
                gid = self.next_global_id
                self.next_global_id += 1
                child_global_ids.append(gid)
            self.level_graph_ids[next_level_idx] = child_global_ids

            parent_global_ids = self.level_graph_ids[level]
            for parent_local_idx, child_local_idx in parent_child_pairs:
                parent_gid = parent_global_ids[parent_local_idx]
                child_gid = child_global_ids[child_local_idx]
                self.edges.append((parent_gid, child_gid))

            if any(g.num_nodes() == 1 for g in next_level):
                print(r"Reached the trivial graph $K_1$. Stopping.")
                break

            level = next_level_idx

        print(f"Finished building spectrum with {len(self.contraction_levels)} levels.")
        print(f"Total unique graphs (global IDs): {self.next_global_id}")
        print(f"Total contraction edges recorded: {len(self.edges)}")

    def save_spectrum_to_json(self, filename: str = "contraction_spectrum.json"):
        filepath = os.path.join(self.output_path, filename)

        spectrum_data: dict = {
            "metadata": {
                "num_levels": len(self.contraction_levels),
                "original_graph_nodes": len(self.graph.nodes)
            },
            "graph_ids": {},
            "edges": [
                {"parent": int(u), "child": int(v)}
                for (u, v) in self.edges
            ],
        }

        for level, graphs in self.contraction_levels.items():
            level_key = f"level_{level}"
            spectrum_data[level_key] = []
            spectrum_data["graph_ids"][level_key] = [
                int(gid) for gid in self.level_graph_ids.get(level, [])
            ]

            for g in graphs:
                spectrum_data[level_key].append({
                    "nodes": g.nodes,
                    "edges": g.get_edges(),
                    "num_nodes": g.num_nodes(),
                    "adjacency_matrix": g.adj  # already list-of-lists
                })

        with open(filepath, "w") as f:
            json.dump(spectrum_data, f, indent=4)

        print(f"[+] Full contraction spectrum saved to {filepath}")

    def get_spectrum(self):
        return self.contraction_levels

    def get_edges(self):
        return self.edges

    def get_level_graph_ids(self):
        return self.level_graph_ids
