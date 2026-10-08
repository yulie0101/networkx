"""Standalone worker: runs max_weight_matching(maxcardinality=True) on a
graph read from a pickle file, times it, writes (size, elapsed) to an
output pickle file. Run as its own process (via subprocess + timeout) so
a pathologically slow Edmonds call can be killed cleanly without taking
the parent process down with it.

Usage: python _edmonds_worker.py <in_pickle> <out_pickle>
"""

import pickle
import sys
import time

import networkx as nx


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    with open(in_path, "rb") as f:
        nodes, edges = pickle.load(f)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    t0 = time.perf_counter()
    matching = nx.max_weight_matching(G, maxcardinality=True)
    elapsed = time.perf_counter() - t0
    with open(out_path, "wb") as f:
        pickle.dump((len(matching), elapsed), f)


if __name__ == "__main__":
    main()
