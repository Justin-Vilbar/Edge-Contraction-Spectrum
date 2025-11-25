import os
import time
import networkx as nx
from ecs_module import Graph, EdgeContraction, SpectrumPDFVisualizer

def main(G, fileName, output_dir):
    start_time = time.time()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    graphs_dir = os.path.join(current_dir, "graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    output_path = os.path.join(graphs_dir, output_dir)
    os.makedirs(output_path, exist_ok=True)

    print("\n=== Edge Contraction Pipeline ===\n")

    print("Step 1: Generating initial graph...")
    graph_torch = Graph(G)
    graph_file = f"{fileName}.json"
    graph_torch.save_to_json(graph_file, output_dir)
    print(f"✓ Initial graph saved to: {os.path.join(output_path, graph_file)}")

    print("\nStep 2: Generating contraction spectrum...")
    json_input_path = os.path.join(output_path, graph_file)
    ec = EdgeContraction(json_input_path, output_dir)
    ec.build_spectrum()
    spectrum_file = f"{fileName}_spectrum.json"
    spectrum_json_dir = os.path.join(current_dir, "spectrum_json")
    if not os.path.exists(spectrum_json_dir):
        os.makedirs(spectrum_json_dir)
    ec.save_spectrum_to_json(os.path.join(spectrum_json_dir, spectrum_file))
    ec.save_spectrum_to_json(spectrum_file)
    print(f"✓ Contraction spectrum saved to: {os.path.join(output_path, spectrum_file)}")

    print("\nStep 3: Generating visualization...")
    spectrum_json = os.path.join(output_path, spectrum_file)
    output_with_non_iso_graphs_pdf = os.path.join(output_path, f"{fileName}_spectrum_with_non_iso_graphs.pdf")
    output_pdf = os.path.join(output_path, f"{fileName}_spectrum.pdf")
    output_png = os.path.join(output_path, f"{fileName}_spectrum.png")
    visualizer = SpectrumPDFVisualizer(spectrum_json, fileName)

    spectrum_pdf_dir = os.path.join(current_dir, "spectrum_pdf")
    if not os.path.exists(spectrum_pdf_dir):
        os.makedirs(spectrum_pdf_dir)

    spectrum_pdf_path = os.path.join(spectrum_pdf_dir, f"{fileName}_spectrum.pdf")
    visualizer.save_overview_pdf(spectrum_pdf_path)

    pdf_with_non_iso_path = visualizer.save_pdf(output_with_non_iso_graphs_pdf)
    pdf_path = visualizer.save_overview_pdf(output_pdf)
    png_path = visualizer.save_overview_image(output_png)

    print(f"✓ PDF visualization with non-isomorphic graphs saved to: {pdf_with_non_iso_path}")
    print(f"✓ PDF visualization saved to: {pdf_path}")
    print(f"✓ PDF visualization saved to: {png_path}")

    end_time = time.time()
    duration = end_time - start_time
    print(f"\nPipeline completed in {duration:.2f} seconds")
    print("\nOutput files:")
    print(f"1. Graph JSON:     {os.path.join(output_path, graph_file)}")
    print(f"2. Spectrum JSON:  {os.path.join(output_path, spectrum_file)}")
    print(f"3. Spectrum with Non-Isomorphic Graphs PDF:   {pdf_with_non_iso_path}")
    print(f"4. Spectrum PDF:   {pdf_path}")
    print(f"5. Spectrum PNG:   {png_path}")
    print("\nDone! 🎉")

def process_graph_type(graph_type, max_nodes: int):
    i = 3
    while i <= max_nodes:
        if graph_type == "path":
            fileName = f"P_{i}"
            G = nx.path_graph(i)
        elif graph_type == "cycle":
            fileName = f"C_{i}"
            G = nx.cycle_graph(i)
        elif graph_type == "star":
            fileName = f"K_{{1, {i}}}"
            G = nx.star_graph(i)
        elif graph_type == "complete":
            fileName = f"K_{i}"
            G = nx.complete_graph(i)
        elif graph_type == "complete_bipartite_2":
            fileName = f"K_{{2, {i}}}"
            G = nx.complete_bipartite_graph(2, i)
        elif graph_type == "complete_bipartite_3":
            fileName = f"K_{{3, {i}}}"
            G = nx.complete_bipartite_graph(3, i)
        elif graph_type == "fan":
            fileName = f"F_{i}"
            G = Graph.fan_graph(i + 1)
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")
        
        output_dir = fileName
        main(G, fileName, output_dir) 
        i += 1

if __name__ == "__main__":
    print("Processing Path Graphs...")
    process_graph_type("path")

    print("Processing Cycle Graphs...")
    process_graph_type("cycle")

    print("Processing Star Graphs...")
    process_graph_type("star")

    print("Processing Complete Graphs...")
    process_graph_type("complete")

    print("Processing Complete Bipartite (2) Graphs...")
    process_graph_type("complete_bipartite_2")

    print("Processing Complete Bipartite (3) Graphs...")
    process_graph_type("complete_bipartite_3")

    print("Processing Fan Graphs...")
    process_graph_type("fan")