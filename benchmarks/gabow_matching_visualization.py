"""Draws one illustrative graph together with the maximum cardinality
matching found by max_cardinality_matching_gabow, highlighting matched
edges distinctly from unmatched ones.

Standalone script, not collected by pytest and not an asv benchmark:

    python benchmarks/gabow_matching_visualization.py

Saves gabow_matching_example.png next to this script.

The example graph is two odd triangles (an odd cycle in each) joined by a
bridge edge, plus a couple of pendant leaves -- small enough to read at a
glance, but structured so that Phase 1 must actually contract a blossom
around one of the triangles before the search can complete an augmenting
path through it (see the "blossom contraction" tests in
networkx/algorithms/tests/test_matching.py for the same idea used as a
correctness check rather than a picture).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import networkx as nx


def build_example_graph():
    G = nx.Graph()
    # Two odd triangles (blossoms), joined by a bridge edge, each with one
    # pendant leaf hanging off a non-bridge vertex.
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])  # triangle 1
    G.add_edges_from([(3, 4), (4, 5), (5, 3)])  # triangle 2
    G.add_edge(2, 3)  # bridge between the two triangles
    G.add_edge(0, "leaf_a")
    G.add_edge(4, "leaf_b")
    return G


def main():
    G = build_example_graph()
    matching = nx.max_cardinality_matching_gabow(G)
    assert nx.is_matching(G, matching)

    matched_edges = {frozenset(e) for e in matching}
    unmatched_edges = [e for e in G.edges() if frozenset(e) not in matched_edges]
    matched_edges_ordered = [e for e in G.edges() if frozenset(e) in matched_edges]

    pos = nx.spring_layout(G, seed=7)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    nx.draw_networkx_nodes(G, pos, node_color="#dddddd", edgecolors="black", ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax)
    nx.draw_networkx_edges(
        G, pos, edgelist=unmatched_edges, edge_color="#bbbbbb", width=1.5, ax=ax
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=matched_edges_ordered,
        edge_color="crimson",
        width=4.0,
        ax=ax,
    )

    ax.set_title(
        f"max_cardinality_matching_gabow\n{len(matching)} matched edges "
        f"(thick red) out of {G.number_of_edges()} total edges (thin gray)",
        fontsize=12,
    )
    ax.axis("off")
    fig.tight_layout()

    outpath = Path(__file__).with_name("gabow_matching_example.png")
    fig.savefig(outpath, dpi=150)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Matching found ({len(matching)} edges): {sorted(matching)}")
    print(f"Saved plot to {outpath}")


if __name__ == "__main__":
    main()
