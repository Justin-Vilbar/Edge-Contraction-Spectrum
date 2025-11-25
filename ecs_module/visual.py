import json
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import math
from pathlib import Path

class SpectrumPDFVisualizer:
    """
    Visualize a contraction spectrum JSON and export it as a PDF.
    One contraction level per PDF page; multiple graphs per page arranged in a grid.
    """
    def __init__(self, spectrum_json_path: str = None, fileName: str = None):
        spectrum_json_path = str(spectrum_json_path)
        self.spectrum_path = spectrum_json_path
        self.spectrum_data = self._load_spectrum(spectrum_json_path)
        self.fileName = fileName or Path(spectrum_json_path).stem

        self.base_page_width = 11.0      
        self.base_page_height = 8.5      
        self.base_cols = 3               
        self.max_graphs_per_page = 9     
        self.node_size = 700             
        self.edge_width = 2.0
        self.font_size = 10
        self.label_fontweight = 'bold'
        self.layout_seed = 42            
        self.title_fontsize = 16
        self.min_graph_size = 3.0       
        self.page_pad_inches = 0.18
        self.axis_pad_x = 4
        self.axis_pad_y = 1.75

    def _load_spectrum(self, path):
        with open(path, "r") as f:
            data = json.load(f)

        if "metadata" in data:
            self.num_levels = data["metadata"]["num_levels"]
            self.original_nodes = data["metadata"]["original_graph_nodes"]
        return data

    def _build_graph_from_data(self, graph_data):
        """Construct a NetworkX graph from one graph JSON entry."""
        G = nx.Graph()
        
        nodes = [str(n) for n in graph_data.get("nodes", [])]
        G.add_nodes_from(nodes)
        

        edges = graph_data.get("edges", [])
        if edges:
            edge_list = [(str(u), str(v)) for u, v in edges]
            G.add_edges_from(edge_list)
        else:
            adj = graph_data.get("adjacency_matrix", [])
            n = len(nodes)
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i][j] == 1:
                        G.add_edge(nodes[i], nodes[j])
        return G

    def _node_colors_by_degree(self, G):
        """Return a list of node colors (mapped from degree) to improve visibility."""
        degrees = dict(G.degree())
        if not degrees:
            return []
        max_deg = max(degrees.values()) or 1

        norm = [(degrees[n] / max_deg) for n in G.nodes()]
        cmap = plt.get_cmap("plasma")  
        return [cmap(x) for x in norm]

    def _compute_positions(self, G):
        """Compute a deterministic layout for the graph (consistent across runs)."""
        if len(G) == 0:
            return {}
        if len(G) <= 3:
            return nx.circular_layout(G)
        try:
            pos = nx.kamada_kawai_layout(G)
        except Exception:
            try:
                k = 2.0/math.sqrt(len(G))  
                pos = nx.spring_layout(G, k=k, seed=self.layout_seed, iterations=200)
            except Exception:
                pos = nx.circular_layout(G)
        return pos

    def _build_spectrum(self):
        """
        Build the edge contraction spectrum data structure using the
        parent to child relationships encoded in the JSON produced by
        EdgeContraction.save_spectrum_to_json.
        """
        spectrum = self.spectrum_data

        graph_ids_by_level = spectrum.get("graph_ids", {})

        canon_to_snapshot = {}
        all_global_ids = set()

        level_keys = sorted(
            [k for k in spectrum.keys() if k.startswith("level_")],
            key=lambda k: int(k.split("_")[1])
        )

        for level_key in level_keys:
            graphs = spectrum[level_key]              
            level_ids = graph_ids_by_level.get(level_key, [])

            for gid, graph_data in zip(level_ids, graphs):
                nodes = graph_data["nodes"]
                adj_matrix = graph_data["adjacency_matrix"]

                adj_dict = {}
                for i, node in enumerate(nodes):
                    neighbors = {
                        nodes[j]
                        for j, is_connected in enumerate(adj_matrix[i])
                        if is_connected == 1
                    }
                    adj_dict[node] = neighbors

                canon_to_snapshot[gid] = adj_dict
                all_global_ids.add(gid)

        canon_list = sorted(all_global_ids)
        id_to_idx = {gid: idx for idx, gid in enumerate(canon_list)}

        edges_json = spectrum.get("edges", [])
        edges = []
        for e in edges_json:
            parent_gid = e["parent"]
            child_gid = e["child"]

            if parent_gid in id_to_idx and child_gid in id_to_idx:
                u = id_to_idx[parent_gid]
                v = id_to_idx[child_gid]
                edges.append((u, v))

        return canon_list, canon_to_snapshot, edges


    def _set_zorder(self, artist, z):
        """Set zorder on a single Matplotlib collection or an iterable of them."""
        if artist is None:
            return
        if isinstance(artist, (list, tuple)):
            for a in artist:
                try:
                    a.set_zorder(z)
                except Exception:
                    pass
        else:
            try:
                artist.set_zorder(z)
            except Exception:
                pass

    def _set_clip_on(self, artist, flag: bool):
        """Set clip_on on a single collection or an iterable of them."""
        if artist is None:
            return
        if isinstance(artist, (list, tuple)):
            for a in artist:
                try:
                    a.set_clip_on(flag)
                except Exception:
                    pass
        else:
            try:
                artist.set_clip_on(flag)
            except Exception:
                pass


    def _draw_spectrum_digraph(self, canon_list, canon_to_snapshot, edges):
        """Draw the directed graph visualization of the contraction spectrum (page 2)."""
        DG = nx.DiGraph()
        DG.add_nodes_from(range(len(canon_list)))
        DG.add_edges_from(edges)

        step_indices = {}
        layers = []
        for idx, canon in enumerate(canon_list):
            k = len(canon_to_snapshot[canon])
            step_indices[idx] = k
            if k not in layers:
                layers.append(k)
        layers = sorted(layers, reverse=True)
        layer_nodes = {L: [] for L in layers}
        for idx, L in step_indices.items():
            layer_nodes[L].append(idx)

        x_gap, y_gap = 7.5, 5
        fig_width  = max(self.base_page_width,  x_gap * max(len(v) for v in layer_nodes.values()) + 3.5)
        fig_height = max(self.base_page_height, y_gap * len(layers))
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        pos = {}
        for i, L in enumerate(layers):
            nodes = layer_nodes[L]
            y = fig_height - (i * y_gap)
            n = len(nodes)
            if n == 1:
                pos[nodes[0]] = (fig_width / 2.0, y+2)
            else:
                span = x_gap * (n - 1)
                x0 = fig_width / 2.0 - span / 2.0
                for j, node in enumerate(nodes):
                    pos[node] = (x0 + j * x_gap, y+3)

        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        x_mid = (max(xs) + min(xs)) / 2.0
        y_mid = (max(ys) + min(ys)) / 2.0
        for k, (x, y) in list(pos.items()):
            pos[k] = (x - x_mid, y - y_mid)

        extra_y_space = 0.0  
        ax.set_xlim(min(xs) - self.axis_pad_x - x_mid, max(xs) + self.axis_pad_x - x_mid)
        ax.set_ylim(min(ys) - self.axis_pad_y - y_mid - extra_y_space,
                    max(ys) + self.axis_pad_y - y_mid + extra_y_space)
        ax.set_aspect('equal', adjustable='box')
        ax.margins(0.0)

        edge_colls = nx.draw_networkx_edges(
            DG, pos, ax=ax,
            arrows=True,
            arrowstyle='-|>',
            arrowsize=40,
            min_source_margin=40,
            min_target_margin=100,
            connectionstyle="arc3,rad=0.07",
            width=2.2,
            edge_color='#34495E'
        )
        self._set_zorder(edge_colls, 2)

        node_coll = nx.draw_networkx_nodes(
            DG, pos, ax=ax,
            node_color="#ffffff",
            node_size=38000,
            edgecolors="black", linewidths=1.6
        )
        try: node_coll.set_zorder(3)
        except: pass

        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        inset_size_in = 2.2
        for idx in DG.nodes:
            snap = canon_to_snapshot[canon_list[idx]]
            g = self._build_graph_from_data({
                "nodes": list(snap.keys()),
                "adjacency_matrix": None,
                "edges": self._get_edges_from_dict(snap),
            })
            x, y = pos[idx]
            ax_inset = inset_axes(
                ax, width=inset_size_in, height=inset_size_in,
                loc='center', bbox_to_anchor=(x, y),
                bbox_transform=ax.transData, borderpad=0
            )
            ax_inset.patch.set_facecolor('none')
            ax_inset.patch.set_alpha(0.0)
            ax_inset.set_clip_on(False)
            try: ax_inset.set_zorder(2.5)
            except: pass

            pos_inset = self._compute_positions(g)
            if pos_inset:
                xs_i = [p[0] for p in pos_inset.values()]
                ys_i = [p[1] for p in pos_inset.values()]
                xmin, xmax = min(xs_i), max(xs_i)
                ymin, ymax = min(ys_i), max(ys_i)
                span = max(xmax - xmin, ymax - ymin)
                pad = 0.25 if span == 0 else 0.20 * span
                ax_inset.set_xlim(xmin - pad, xmax + pad)
                ax_inset.set_ylim(ymin - pad, ymax + pad)
            ax_inset.set_aspect('equal', adjustable='box')
            ax_inset.margins(0.05)
            ax_inset.set_axis_off()

            pc = nx.draw_networkx_nodes(
                g, pos_inset, ax=ax_inset,
                node_color='#90caf9', node_size=200, linewidths=1.8, edgecolors='black'
            )
            lc = nx.draw_networkx_edges(g, pos_inset, ax=ax_inset, width=2.0, edge_color='#34495E')
            self._set_clip_on(pc, False)
            self._set_clip_on(lc, False)
            #nx.draw_networkx_labels(g, pos_inset, ax=ax_inset, font_size=7, font_weight="bold")

        outline = nx.draw_networkx_nodes(
            DG, pos, ax=ax, node_color='none',
            node_size=38000, edgecolors='black', linewidths=1.6
        )
        try: outline.set_zorder(10)
        except: pass

        for i, idx in enumerate(DG.nodes):
            x, y = pos[idx]
            #ax.text(x, y + 2.0, f"G{idx+1}", ha='center', va='center',
                    #fontsize=12, zorder=11)

        for i in range(len(layers) - 1):
            y_data = (fig_height - ((i + 3) * y_gap)) - y_mid
            y_axes = (y_data - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
            #ax.text(0.01, y_axes, f"Level {i+1}",
                    #transform=ax.transAxes, ha='left', va='center',
                    #fontsize=12, fontweight='bold', zorder=11)

        #ax.set_title(rf"Edge Contraction Spectrum $\Sigma({self.fileName})$", fontsize=60)
        ax.set_axis_off()
        return fig

    def save_overview_image(self, output_image: str = "contraction_spectrum_overview.png",
                        dpi: int = 300, transparent: bool = False, pad_inches: float = 0.4):
        """
        Save the page-2 overview (edge contraction spectrum digraph with insets) as a PNG/JPEG/etc.
        """
        out_path = Path(output_image).resolve()
        canon_list, canon_to_snapshot, edges = self._build_spectrum()
        fig = self._draw_spectrum_digraph(canon_list, canon_to_snapshot, edges)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches, transparent=transparent)
        plt.close(fig)
        print(f"[+] Overview image saved to: {out_path}")
        return str(out_path)


    def save_level_images(self, output_dir: str = "spectrum_levels_imgs",
                        dpi: int = 300, transparent: bool = False, pad_inches: float = 0.2):
        """
        Save each contraction level page (the grid of graphs) as its own image.
        Images are named level_0.png, level_1.png, ...
        """
        out_dir = Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        level_keys = [k for k in self.spectrum_data.keys() if k.startswith("level_")]
        sorted_keys = sorted(level_keys, key=lambda k: int(k.split("_")[1]))

        for level_key in sorted_keys:
            graphs = self.spectrum_data[level_key]
            if not graphs:
                fig = plt.figure(figsize=(self.base_page_width, self.base_page_height))
                fig.suptitle(f"{level_key} (no graphs)", fontsize=self.title_fontsize)
                plt.axis("off")
                out_path = out_dir / f"{level_key}.png"
                fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches, transparent=transparent)
                plt.close(fig)
                continue

            num_graphs = len(graphs)
            if num_graphs <= self.max_graphs_per_page:
                cols = min(self.base_cols, num_graphs)
                page_size = (self.base_page_width, self.base_page_height)
            else:
                max_cols = max(3, math.floor(self.base_page_width / self.min_graph_size))
                cols = min(max_cols, math.ceil(math.sqrt(num_graphs)))
                rows = math.ceil(num_graphs / cols)
                graphs_width = cols * self.min_graph_size
                graphs_height = rows * self.min_graph_size
                page_size = (
                    max(self.base_page_width, 1.1 * graphs_width),
                    max(self.base_page_height, 1.2 * graphs_height),
                )
            rows = math.ceil(num_graphs / cols)

            fig, axes = plt.subplots(rows, cols, figsize=page_size)
            fig.suptitle(f"{level_key.replace('_', ' ').title()}",
                        fontsize=self.title_fontsize, fontweight="bold", y=0.98)

            if isinstance(axes, plt.Axes):
                axes = [[axes]]
            elif rows == 1:
                axes = [axes]
            axes_flat = []
            for r in range(rows):
                row_axes = list(axes[r]) if hasattr(axes[r], "__iter__") else [axes[r]]
                while len(row_axes) < cols:
                    row_axes.append(None)
                axes_flat.extend(row_axes)

            for idx, graph_data in enumerate(graphs):
                ax = axes_flat[idx]
                if ax is None:
                    continue
                G = self._build_graph_from_data(graph_data)
                pos = self._compute_positions(G)
                node_colors = self._node_colors_by_degree(G)
                scale = 1.0 if num_graphs <= self.max_graphs_per_page else 0.7
                ax.set_axis_off()
                nx.draw_networkx_nodes(G, pos, ax=ax,
                                    node_color=node_colors if node_colors else "lightblue",
                                    node_size=self.node_size * scale,
                                    linewidths=1.5, edgecolors="black")
                nx.draw_networkx_edges(G, pos, ax=ax, width=self.edge_width, alpha=0.9)
                nx.draw_networkx_labels(G, pos, ax=ax,
                                        font_size=self.font_size * scale,
                                        font_weight=self.label_fontweight)
                ax.set_title(f"{level_key.replace('_', ' ').title()} — Graph {idx + 1}",
                            fontsize=10 if num_graphs <= self.max_graphs_per_page else 8,
                            pad=6, fontweight="bold")

            for extra_ax in axes_flat[num_graphs:]:
                if extra_ax is not None:
                    extra_ax.set_axis_off()

            out_path = out_dir / f"{level_key}.png"
            plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches, transparent=transparent)
            plt.close(fig)
            print(f"[+] Level image saved: {out_path}")

            return str(out_dir)


    def _get_edges_from_dict(self, graph_dict):
        """Helper to extract edges from adjacency dictionary."""
        edges = []
        seen = set()
        for v1, neighbors in graph_dict.items():
            for v2 in neighbors:
                edge = tuple(sorted([v1, v2]))
                if edge not in seen:
                    edges.append(edge)
                    seen.add(edge)
        return edges

    def save_pdf(self, output_pdf="contraction_spectrum.pdf"):
        """Render the full spectrum into a single PDF (one page per level)."""
        out_path = Path(output_pdf).resolve()
        with PdfPages(out_path) as pdf:
            canon_list, canon_to_snapshot, edges = self._build_spectrum()
            spectrum_fig = self._draw_spectrum_digraph(canon_list, canon_to_snapshot, edges)
            pdf.savefig(spectrum_fig, bbox_inches='tight', pad_inches=0.4)
            plt.close(spectrum_fig)

            level_keys = [k for k in self.spectrum_data.keys() if k.startswith("level_")]
            sorted_keys = sorted(level_keys, key=lambda k: int(k.split("_")[1]))

            for level_key in sorted_keys:
                graphs = self.spectrum_data[level_key]
                if not graphs:
                    fig = plt.figure(figsize=(self.base_page_width, self.base_page_height))
                    fig.suptitle(f"{level_key} (no graphs)", fontsize=self.title_fontsize)
                    plt.axis('off')
                    pdf.savefig(fig)
                    plt.close(fig)
                    continue

                num_graphs = len(graphs)

                if num_graphs <= self.max_graphs_per_page:
                    cols = min(self.base_cols, num_graphs)
                    page_size = (self.base_page_width, self.base_page_height)
                else:
                    max_cols = max(3, math.floor(self.base_page_width / self.min_graph_size))
                    cols = min(max_cols, math.ceil(math.sqrt(num_graphs)))
                    
                    graphs_width = cols * self.min_graph_size
                    width = max(self.base_page_width, graphs_width * 1.1)  
                    
                    rows = math.ceil(num_graphs / cols)
                    graphs_height = rows * self.min_graph_size
                    height = max(self.base_page_height, graphs_height * 1.2)  
                    
                    page_size = (width, height)

                rows = math.ceil(num_graphs / cols)

                fig, axes = plt.subplots(rows, cols, figsize=page_size)
                fig.suptitle(f"{level_key.replace('_', ' ').title()}", fontsize=self.title_fontsize, fontweight='bold', y=0.98)
                if isinstance(axes, plt.Axes):
                    axes = [[axes]]
                elif rows == 1:
                    axes = [axes]
                axes_flat = []
                for r in range(rows):
                    if cols == 1:
                        axes_row = [axes[r][0] if isinstance(axes[r], (list, tuple)) else axes[r]]
                    else:
                        axes_row = list(axes[r]) if hasattr(axes[r], "__iter__") else [axes[r]]
                    while len(axes_row) < cols:
                        axes_row.append(None)
                    axes_flat.extend(axes_row)

                for idx, graph_data in enumerate(graphs):
                    ax = axes_flat[idx]
                    if ax is None:
                        continue

                    G = self._build_graph_from_data(graph_data)
                    pos = self._compute_positions(G)
                    node_colors = self._node_colors_by_degree(G)

                    scale_factor = 1.0 if num_graphs <= self.max_graphs_per_page else 0.7
                    adjusted_node_size = self.node_size * scale_factor
                    adjusted_font_size = self.font_size * scale_factor
                    
                    ax.set_axis_off()
                    nx.draw_networkx_nodes(G, pos, ax=ax,
                                           node_color=node_colors if node_colors else "lightblue",
                                           node_size=adjusted_node_size,
                                           linewidths=1.5,
                                           edgecolors="black")
                    nx.draw_networkx_edges(G, pos, ax=ax, width=self.edge_width, alpha=0.9)
                    nx.draw_networkx_labels(G, pos, ax=ax, 
                                          font_size=adjusted_font_size, 
                                          font_weight=self.label_fontweight)

                    graph_title = f"{level_key.replace('_', ' ').title()} — Graph {idx + 1}"
                    title_size = 10 if num_graphs <= self.max_graphs_per_page else 8
                    ax.set_title(graph_title, fontsize=title_size, pad=6, fontweight='bold')

                for extra_ax in axes_flat[num_graphs:]:
                    if extra_ax is not None:
                        extra_ax.set_axis_off()

                plt.tight_layout(rect=[0, 0, 1, 0.95])
                pdf.savefig(fig)
                plt.close(fig)

        print(f"[+] PDF saved to: {out_path}")
        return str(out_path)
    
    def save_overview_pdf(self, output_pdf: str = "edge_contraction_spectrum.pdf",
                      pad_inches: float = 0.4):
        """
        Save ONLY the overview (page 2) of the contraction spectrum as a single-page PDF.
        This is useful for thesis figures or publication inserts.
        """
        out_path = Path(output_pdf).resolve()

        canon_list, canon_to_snapshot, edges = self._build_spectrum()

        fig = self._draw_spectrum_digraph(canon_list, canon_to_snapshot, edges)

        fig.savefig(out_path, bbox_inches='tight', pad_inches=pad_inches)
        plt.close(fig)

        print(f"[+] Overview-only PDF saved to: {out_path}")
        return str(out_path)