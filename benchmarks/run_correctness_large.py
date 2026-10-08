"""Task-3A: correctness of max_cardinality_matching_gabow on >= 3,000 large
random graphs, n from 200 to 3,000, densities sparse (m=3n) to dense
(p=0.5, n<=1000 only -- Edmonds is O(n**3)), 3 node orders, greedy on/off.
Checks: valid matching (nx.is_matching) and size == max_weight_matching
(maxcardinality=True).

RESUMABLE: appends each finished run to benchmarks/results/
correctness_large.csv immediately; on restart, skips (n, density, seed,
order, greedy) combos already in the CSV. Safe to kill and rerun anytime.

One Edmonds (reference) call per (n, density, seed) -- NOT per order/greedy,
since the reference matching's SIZE does not depend on node order or which
algorithm/greedy setting produced the candidate; this is cached and reused
across that group's 6 (order, greedy) combos, a 6x reduction in the
(by far) most expensive calls.

Usage:
    python benchmarks/run_correctness_large.py            # run to completion
    python benchmarks/run_correctness_large.py --report    # just print the
                                                             # table from the
                                                             # existing CSV
"""

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import networkx as nx

OUTDIR = Path(__file__).with_name("results")
OUTDIR.mkdir(parents=True, exist_ok=True)
CSV_NAME = "correctness_large.csv"
FIELDS = [
    "n",
    "actual_n",
    "actual_m",
    "density",
    "seed",
    "order",
    "greedy",
    "valid",
    "gabow_size",
    "ref_size",
    "ok",
]

N_SMALL = [200, 300, 400, 500, 600, 800, 1000]
N_LARGE = [1200, 1500, 1800, 2100, 2400, 2700, 3000]
SEEDS = list(range(18))
ORDERS = ("ascending", "descending", "random")


def build_base_graph(n, density, seed):
    if density == "sparse":
        m = 3 * n
        return nx.gnm_random_graph(n, m, seed=seed)
    if density == "medium":
        return nx.gnp_random_graph(n, 0.1, seed=seed)
    if density == "dense":
        return nx.gnp_random_graph(n, 0.5, seed=seed)
    raise ValueError(density)


def ordered_graph(base, order, rng):
    nodes = list(base.nodes())
    if order == "ascending":
        node_order = sorted(nodes)
    elif order == "descending":
        node_order = sorted(nodes, reverse=True)
    else:
        node_order = nodes[:]
        rng.shuffle(node_order)
    G = nx.Graph()
    G.add_nodes_from(node_order)
    G.add_edges_from(base.edges())
    return G


def _append_row(row):
    path = OUTDIR / CSV_NAME
    exists = path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)


def _load_done():
    path = OUTDIR / CSV_NAME
    if not path.exists():
        return set()
    done = set()
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            done.add((r["n"], r["density"], r["seed"], r["order"], r["greedy"]))
    return done


def configs():
    for n in N_SMALL:
        for density in ("sparse", "medium", "dense"):
            for seed in SEEDS:
                yield n, density, seed
    for n in N_LARGE:
        for seed in SEEDS:
            yield n, "sparse", seed


def run():
    done = _load_done()
    print(f"{len(done)} runs already done, will be skipped", flush=True)
    t0 = time.perf_counter()
    n_new = 0
    n_total_configs = 0
    for n, density, seed in configs():
        n_total_configs += 1
        keys = [
            (str(n), density, str(seed), order, str(g))
            for order in ORDERS
            for g in (True, False)
        ]
        if all(k in done for k in keys):
            continue
        base = build_base_graph(n, density, seed)
        ref_size = len(nx.max_weight_matching(base, maxcardinality=True))
        rng = random.Random(seed * 1000003 + hash(density) % 97)
        for order in ORDERS:
            G = ordered_graph(base, order, rng)
            for greedy in (True, False):
                key = (str(n), density, str(seed), order, str(greedy))
                if key in done:
                    continue
                got = nx.max_cardinality_matching_gabow(G, _skip_greedy_init=not greedy)
                valid = nx.is_matching(G, got)
                ok = valid and (len(got) == ref_size)
                row = {
                    "n": n,
                    "actual_n": G.number_of_nodes(),
                    "actual_m": G.number_of_edges(),
                    "density": density,
                    "seed": seed,
                    "order": order,
                    "greedy": greedy,
                    "valid": valid,
                    "gabow_size": len(got),
                    "ref_size": ref_size,
                    "ok": ok,
                }
                _append_row(row)
                done.add(key)
                n_new += 1
                if not ok:
                    print(
                        f"FAILURE: n={n} density={density} seed={seed} order={order} greedy={greedy} "
                        f"valid={valid} gabow={len(got)} ref={ref_size}",
                        flush=True,
                    )
        if n_total_configs % 10 == 0:
            print(
                f"...config {n_total_configs}, {n_new} new rows this run, "
                f"{time.perf_counter() - t0:.1f}s elapsed",
                flush=True,
            )
    print(f"DONE: {n_new} new rows, {time.perf_counter() - t0:.1f}s total", flush=True)


def report():
    path = OUTDIR / CSV_NAME
    if not path.exists():
        print("no data yet")
        return
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"total rows: {len(rows)}")
    # n range | density | graphs | failures
    buckets = {}
    for r in rows:
        n = int(r["n"])
        bucket = "200-1000" if n <= 1000 else "1200-3000"
        key = (bucket, r["density"])
        buckets.setdefault(key, [0, 0])
        buckets[key][0] += 1
        if r["ok"] != "True":
            buckets[key][1] += 1
    print(f"{'n range':12} {'density':8} {'graphs':8} {'failures':8}")
    for (bucket, density), (count, fails) in sorted(buckets.items()):
        print(f"{bucket:12} {density:8} {count:8} {fails:8}")
    total_fail = sum(1 for r in rows if r["ok"] != "True")
    print(f"\nTOTAL: {len(rows)} runs, {total_fail} failures")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--report", action="store_true")
    args = p.parse_args()
    if args.report:
        report()
    else:
        run()
        report()
