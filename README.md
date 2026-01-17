# Edge-Contraction-Spectrum

[![DOI](https://zenodo.org/badge/1094065327.svg)](https://doi.org/10.5281/zenodo.17602990)

This project implements a automated pipeline for generating and visualizing the **Edge Contraction Spectrum** of various graph classes. The pipeline iteratively contracts edges in a graph, filters for non-isomorphic results at each stage, and maps the relationships between parent and child graphs until the trivial graph  is reached.

---

## Pipeline Overview

The process follows a three-step sequence to transform a standard graph into a visual hierarchy:

1. **Graph Initialization**: A starting graph is generated (e.g., a Path, Cycle, or Complete graph) and stored with its adjacency matrix.
2. **Spectrum Generation**: The system performs all possible edge contractions. It uses isomorphism checking to ensure that each level of the spectrum only contains unique graph structures.
3. **Visualization**: The resulting "family tree" of graphs is rendered into a directed graph (digraph). Each node in this digraph is an inset image of the graph at that specific contraction stage.

---

## Module Descriptions

### graph.py

A lightweight wrapper around the `networkx` library. It manages the fundamental graph data:

* Maintains a fixed node ordering and a 0/1 adjacency matrix.
* Provides utility functions for saving graph data to JSON.
* Includes a specialized method for generating **Fan Graphs** .

### edge_contraction.py

The core engine of the pipeline. This module handles the iterative contraction logic:

* **Isomorphism Filtering**: Uses `nx.is_isomorphic` to ensure only distinct graphs are kept at each contraction level.
* **Lineage Tracking**: Records parent-to-child relationships between graphs across levels.
* **Spectrum Export**: Saves the entire hierarchy, including metadata and graph IDs, into a structured JSON format.

### visual.py

The visualization suite used to interpret the spectrum data:

* **Contraction Digraph**: Generates a high-level view showing how graphs evolve through contractions.
* **Inset Rendering**: Embeds small visual representations of the actual graphs within the nodes of the spectrum tree.
* **Multi-format Export**: Produces high-resolution PNGs and multi-page PDFs that detail every non-isomorphic graph discovered.

### main.py

The execution script that orchestrates the processing of multiple graph families. It iterates through:

* Path Graphs 
* Cycle Graphs 
* Star Graphs 
* Complete Graphs 
* Complete Bipartite Graphs  and 
* Fan Graphs 

---

## Output Structure

Upon completion, the pipeline organizes results into specific directories:

| File Type | Location | Description |
| --- | --- | --- |
| **Initial Graph JSON** | `/graphs/[graph_name]/` | The raw data of the starting graph. |
| **Spectrum JSON** | `/spectrum_json/` | The full hierarchy and relationship data. |
| **Spectrum Overview PDF** | `/spectrum_pdf/` | A single-page PDF showing the contraction tree. |
| **Detailed PDF** | `/graphs/[graph_name]/` | A PDF containing every unique graph at every level. |
| **Overview Image** | `/graphs/[graph_name]/` | A PNG version of the spectrum for quick reference. |

---

## Technical Specifications

* **Language**: Python 3.x
* **Primary Libraries**: `networkx`, `matplotlib`, `json`
* **Layout Engines**: Uses `kamada_kawai_layout` and `spring_layout` for deterministic and readable graph visualizations.
* **Termination Condition**: The process stops automatically when a graph reaches 1 node () or a user-defined maximum level.
