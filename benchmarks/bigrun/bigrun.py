"""Portable, standalone benchmark runner for max_cardinality_matching_gabow,
for a machine with more RAM/cores than the dev machine this was built on.
Self-contained: does not import anything from the rest of this repo (run.bat
installs networkx from our branch via pip; that's the only dependency,
plus matplotlib for make_plots.py).

Two experiments, both config-driven (config.json) and resumable (each
finished run is appended to its CSV immediately; a restart skips any
(key) combination already present):

  1. Correctness: Gabow vs max_weight_matching(maxcardinality=True) on
     large random graphs, 3 node orders, greedy on/off.
  2. Complexity/wall-clock: op counts (via _counters) and wall-clock time,
     families F1-F6 (same families as benchmarks/run_all.py), Gabow vs
     Edmonds. Edmonds runs in its own subprocess with a timeout (config
     "edmonds_timeout_sec") so a pathologically slow call can be killed
     without taking this process down. Each graph's Edmonds time is the
     MEDIAN of "edmonds_repeats" separate calls (default 3), so large-n
     speedups do not rest on a single noisy sample; Edmonds is skipped
     entirely for graphs with more than "edmonds_max_n" vertices, and
     once a family's Edmonds times out, it's skipped for that family on
     later (resumed) runs.

Uses a process pool (config "max_workers", null = cpu_count() - 1) so
independent (family, n, seed, ...) runs can proceed on separate cores.

Usage:
    python bigrun.py                  # run everything in config.json
    python bigrun.py --report         # just print tables from existing CSVs
"""

import argparse
import csv
import json
import math
import multiprocessing
import os
import pickle
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import networkx as nx

HERE = Path(__file__).parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
CONFIG_PATH = HERE / "config.json"
EDMONDS_WORKER = HERE / "_edmonds_worker.py"

CORRECTNESS_CSV = "bigrun_correctness.csv"
CORRECTNESS_FIELDS = [
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
    "ref_timed_out",
    "ok",
]

COMPLEXITY_CSV = "bigrun_complexity.csv"
COMPLEXITY_FIELDS = [
    "family",
    "n",
    "n_param",
    "m",
    "seed",
    "algorithm",
    "variant",
    "runtime_sec",
    "iterations",
    "total_ops",
    "timed_out",
    "edmonds_repeats",
    "runtime_min",
    "runtime_max",
]


# ---------------------------------------------------------------------------
# Graph families (same shapes as benchmarks/run_all.py / verify_complexity.py,
# reimplemented here so this file has zero dependency on the rest of the repo)
# ---------------------------------------------------------------------------


def f1_sparse(n, c, seed):
    return nx.gnm_random_graph(n, c * n, seed=seed)


def f2_dense(n, density, seed):
    max_m = n * (n - 1) // 2
    m = max(n - 1, min(max_m, int(density * max_m)))
    return nx.gnm_random_graph(n, m, seed=seed)


def f3_bad_greedy(k, seed=0):
    G = nx.Graph()
    for i in range(k):
        b = 4 * i
        G.add_edge(b + 2, b + 3)
        G.add_edge(b + 1, b + 2)
        G.add_edge(b + 3, b + 4)
    return G


def f4_nested_blossoms(depth, seed=0):
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    tip = 0
    next_id = 3
    for _ in range(depth - 1):
        a, b = next_id, next_id + 1
        G.add_edges_from([(tip, a), (a, b), (b, tip)])
        tip = b
        next_id += 2
    return G


def _complete_g(m_target):
    n_clique = max(2, int(math.sqrt(m_target)))
    G = nx.complete_graph(n_clique)
    return G, 0, n_clique


def _add_chain(G, k, z, next_id):
    a = list(range(next_id, next_id + k))
    G.add_edge(a[0], z)
    for i in range(1, k - 1):
        G.add_edge(a[i], a[i + 1])
        G.add_edge(a[i], z)
    G.add_edge(a[0], a[1])
    return next_id + k


def f5_chains(n_param, seed=0, mode=1):
    m_param = 4 * n_param
    G, z, _n_clique = _complete_g(m_param)
    next_id = G.number_of_nodes()
    for _j in range(n_param // 8):
        next_id = _add_chain(G, 8, z, next_id)
    if mode == 1:
        k = 5
        while k < math.sqrt(n_param):
            next_id = _add_chain(G, 2 * k, z, next_id)
            k += 1
    return G


def f5_true_edge_order(n_param, mode=1):
    """The TRUE edge-insertion order of f5_chains(n_param, mode) -- NOT
    G.edges(), which is node-major (a late edge touching the early,
    high-degree hub z gets pulled forward). The C++ reference's init()
    scans edges in insertion order; feeding it G.edges() instead yields a
    different greedy matching that pre-solves this family in 1 iteration.
    """
    order = []
    n_clique = max(2, int(math.sqrt(4 * n_param)))
    for i in range(n_clique):
        for j in range(i + 1, n_clique):
            order.append((i, j))
    z = 0
    next_id = n_clique

    def add_chain(k):
        nonlocal next_id
        a = list(range(next_id, next_id + k))
        order.append((a[0], z))
        for i in range(1, k - 1):
            order.append((a[i], a[i + 1]))
            order.append((a[i], z))
        order.append((a[0], a[1]))
        next_id += k

    for _j in range(n_param // 8):
        add_chain(8)
    if mode == 1:
        k = 5
        while k < math.sqrt(n_param):
            add_chain(2 * k)
            k += 1
    return order


def f6_long_path(n, seed=0):
    return nx.path_graph(n)


def edge_scan_greedy(edge_order):
    mate = {}
    for u, v in edge_order:
        if u != v and u not in mate and v not in mate:
            mate[u] = v
            mate[v] = u
    return mate


# ---------------------------------------------------------------------------
# Resumable CSV helpers
# ---------------------------------------------------------------------------


def _append_row(name, fields, row):
    path = RESULTS / name
    exists = path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow(row)


def _load_done_keys(name, key_fields):
    path = RESULTS / name
    if not path.exists():
        return set()
    done = set()
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            done.add(tuple(r[k] for k in key_fields))
    return done


def _load_rows(name):
    path = RESULTS / name
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Edmonds via subprocess, with a timeout
# ---------------------------------------------------------------------------


def run_edmonds_with_timeout(G, timeout_sec):
    """Returns (size, elapsed, timed_out). On timeout, (None, timeout_sec, True)."""
    with tempfile.TemporaryDirectory() as td:
        in_path = Path(td) / "in.pkl"
        out_path = Path(td) / "out.pkl"
        with open(in_path, "wb") as f:
            pickle.dump((list(G.nodes()), list(G.edges())), f)
        try:
            subprocess.run(
                [sys.executable, str(EDMONDS_WORKER), str(in_path), str(out_path)],
                timeout=timeout_sec,
                check=True,
                capture_output=True,
            )
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            return None, float(timeout_sec), True
        with open(out_path, "rb") as f:
            size, elapsed = pickle.load(f)
        return size, elapsed, False


# ---------------------------------------------------------------------------
# Environment info
# ---------------------------------------------------------------------------


def env_info():
    try:
        import psutil

        ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        ram_gb = "unknown (install psutil for this)"
    commit = "unknown"
    try:
        import importlib.metadata as im

        dist = im.distribution("networkx")
        direct_url = dist.read_text("direct_url.json")
        if direct_url:
            info = json.loads(direct_url)
            commit = info.get("vcs_info", {}).get("commit_id", "unknown")
    except (ImportError, OSError, ValueError):
        # not installed from a VCS URL, or metadata unreadable: keep "unknown"
        commit = "unknown"
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count_logical": os.cpu_count(),
        "ram_gb": ram_gb,
        "python": platform.python_version(),
        "networkx_version": nx.__version__,
        "networkx_commit": commit,
    }


# ---------------------------------------------------------------------------
# Correctness task
# ---------------------------------------------------------------------------


def _build_base(n, density, seed):
    if density == "sparse":
        return nx.gnm_random_graph(n, 3 * n, seed=seed)
    return nx.gnp_random_graph(n, float(density), seed=seed)


def _ordered(base, order, rng):
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


def correctness_group_task(args):
    """One (n, density, seed) group: one Edmonds reference call (subprocess,
    timeout-capped -- for the large-n correctness sizes this task targets,
    Edmonds will often time out, which is expected and simply skips the
    size/ref_size check for that group, keeping only the is_matching check).
    """
    n, density, seed, orders, greedy_opts, edmonds_timeout = args
    base = _build_base(n, density, seed)
    ref_size, _elapsed, timed_out = run_edmonds_with_timeout(base, edmonds_timeout)
    rng = random.Random(seed * 1000003 + (hash(density) % 97))
    rows = []
    for order in orders:
        G = _ordered(base, order, rng)
        for greedy in greedy_opts:
            got = nx.max_cardinality_matching_gabow(G, _skip_greedy_init=not greedy)
            valid = nx.is_matching(G, got)
            ok = valid if timed_out else (valid and len(got) == ref_size)
            rows.append(
                {
                    "n": n,
                    "actual_n": G.number_of_nodes(),
                    "actual_m": G.number_of_edges(),
                    "density": density,
                    "seed": seed,
                    "order": order,
                    "greedy": greedy,
                    "valid": valid,
                    "gabow_size": len(got),
                    "ref_size": ref_size if not timed_out else "",
                    "ref_timed_out": timed_out,
                    "ok": ok,
                }
            )
    return rows


def run_correctness(cfg, max_workers):
    print("=" * 70)
    print("CORRECTNESS")
    print("=" * 70)
    key_fields = ["n", "density", "seed", "order", "greedy"]
    done = _load_done_keys(CORRECTNESS_CSV, key_fields)
    print(f"{len(done)} runs already done")
    orders = cfg["orders"]
    greedy_opts = cfg["greedy"]
    edmonds_timeout = cfg.get("edmonds_timeout_sec", 60)

    groups = []
    for n in cfg["n_small"]:
        for density in ("sparse", 0.1, 0.5):
            for seed in cfg["seeds"]:
                groups.append((n, density, seed))
    for n in cfg["n_large"]:
        for seed in cfg["seeds"]:
            groups.append((n, "sparse", seed))

    pending = []
    for n, density, seed in groups:
        keys = [
            (str(n), str(density), str(seed), o, str(g))
            for o in orders
            for g in greedy_opts
        ]
        if all(k in done for k in keys):
            continue
        pending.append((n, density, seed, orders, greedy_opts, edmonds_timeout))

    print(
        f"{len(pending)} groups to run ({len(pending) * len(orders) * len(greedy_opts)} rows)"
    )
    t0 = time.perf_counter()
    n_done = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(correctness_group_task, args): args for args in pending}
        for fut in as_completed(futures):
            n, density, seed = futures[fut][:3]
            try:
                rows = fut.result()
            except Exception as e:  # noqa: BLE001 -- one failed task must not abort the pool; it has no row, so a rerun retries it
                print(f"ERROR n={n} density={density} seed={seed}: {e}")
                continue
            for row in rows:
                _append_row(CORRECTNESS_CSV, CORRECTNESS_FIELDS, row)
                if not row["ok"]:
                    print(f"FAILURE: {row}")
            n_done += 1
            if n_done % 5 == 0:
                print(
                    f"...{n_done}/{len(pending)} groups, {time.perf_counter() - t0:.1f}s"
                )
    print(f"Correctness done: {time.perf_counter() - t0:.1f}s")


def report_correctness():
    rows = _load_rows(CORRECTNESS_CSV)
    if not rows:
        print("no correctness data yet")
        return
    buckets = {}
    for r in rows:
        n = int(r["n"])
        bucket = "<=1000" if n <= 1000 else ">1000"
        key = (bucket, r["density"])
        buckets.setdefault(key, [0, 0, 0])
        buckets[key][0] += 1
        if r["ok"] != "True":
            buckets[key][1] += 1
        if r["ref_timed_out"] == "True":
            buckets[key][2] += 1
    print(
        f"{'n range':10} {'density':8} {'graphs':8} {'failures':9} {'edmonds_timeout':15}"
    )
    for (bucket, density), (count, fails, timeouts) in sorted(buckets.items()):
        print(f"{bucket:10} {density!s:8} {count:8} {fails:9} {timeouts:15}")


# ---------------------------------------------------------------------------
# Complexity/wall-clock task
# ---------------------------------------------------------------------------

FAMILY_BUILDERS = {
    "F1_sparse": lambda n, seed, c: f1_sparse(n, c, seed),
    "F2_dense": lambda n, seed, d: f2_dense(n, d, seed),
    "F3_bad_greedy": lambda n, seed: f3_bad_greedy(max(1, n // 4), seed),
    "F4_nested_blossoms": lambda n, seed: f4_nested_blossoms(
        max(1, (n - 1) // 2), seed
    ),
    "F5_chains": lambda n, seed: f5_chains(n, seed),
    "F6_long_path": lambda n, seed: f6_long_path(n, seed),
}


def _expected_vertices(builder_key, n):
    """Vertex count of FAMILY_BUILDERS[builder_key](n, ...) without
    building it (used to apply edmonds_max_n before scheduling)."""
    if builder_key == "F3_bad_greedy":
        return 4 * max(1, n // 4)
    if builder_key == "F4_nested_blossoms":
        return 2 * max(1, (n - 1) // 2) + 1
    if builder_key == "F5_chains":
        n_clique = max(2, int(math.sqrt(4 * n)))
        k_max = math.ceil(math.sqrt(n))
        return (
            n_clique
            + 8 * (n // 8)
            + sum(2 * k for k in range(5, k_max) if k < math.sqrt(n))
        )
    return n


OP_KEYS = [
    "edge_scans",
    "search_steps",
    "blossom_shrink_vertex_steps",
    "uf_base_find_hops",
    "uf_dbase_find_hops",
    "uf_base_union_calls",
    "uf_dbase_union_calls",
    "augmentations",
]


def complexity_task(args):
    (
        family,
        builder_key,
        n,
        n_param,
        seed,
        builder_args,
        edmonds_timeout,
        want_edmonds,
        edmonds_repeats,
    ) = args
    G = FAMILY_BUILDERS[builder_key](n, seed, *builder_args)
    rows = []
    variants = ["greedy_on", "greedy_off"]
    if builder_key == "F5_chains":
        variants.append("edgescan")
    for variant in variants:
        counters = {}
        if variant == "greedy_on":
            kwargs = {"_skip_greedy_init": False}
        elif variant == "greedy_off":
            kwargs = {"_skip_greedy_init": True}
        else:
            order = f5_true_edge_order(n_param)
            assert {frozenset(e) for e in order} == {frozenset(e) for e in G.edges()}
            kwargs = {
                "_skip_greedy_init": True,
                "_initial_mate": edge_scan_greedy(order),
            }
        t0 = time.perf_counter()
        got = nx.max_cardinality_matching_gabow(G.copy(), _counters=counters, **kwargs)
        dt = time.perf_counter() - t0
        total_ops = sum(counters[k] for k in OP_KEYS)
        rows.append(
            {
                "family": family,
                "n": G.number_of_nodes(),
                "n_param": n_param,
                "m": G.number_of_edges(),
                "seed": seed,
                "algorithm": "gabow",
                "variant": variant,
                "runtime_sec": dt,
                "iterations": counters["iterations"],
                "total_ops": total_ops,
                "timed_out": False,
            }
        )
    if want_edmonds:
        # Median of `edmonds_repeats` separate calls; stop at the first
        # timeout (the graph is then marked timed_out only if no call
        # finished at all).
        times, size = [], None
        for _rep in range(edmonds_repeats):
            size_i, dt, timed_out = run_edmonds_with_timeout(G, edmonds_timeout)
            if timed_out:
                break
            size = size_i
            times.append(dt)
        timed_out = not times
        rows.append(
            {
                "family": family,
                "n": G.number_of_nodes(),
                "n_param": n_param,
                "m": G.number_of_edges(),
                "seed": seed,
                "algorithm": "edmonds",
                "variant": "none",
                "runtime_sec": statistics.median(times)
                if times
                else float(edmonds_timeout),
                "iterations": None,
                "total_ops": None,
                "timed_out": timed_out,
                "edmonds_repeats": len(times),
                "runtime_min": min(times) if times else None,
                "runtime_max": max(times) if times else None,
            }
        )
        if not timed_out and size != len(got):
            raise AssertionError(
                f"MISMATCH {family} n={n} seed={seed}: gabow={len(got)} edmonds={size}"
            )
    return rows


def run_complexity(cfg, max_workers):
    print("=" * 70)
    print("COMPLEXITY / WALL-CLOCK")
    print("=" * 70)
    key_fields = ["family", "n_param", "seed", "algorithm", "variant"]
    done = _load_done_keys(COMPLEXITY_CSV, key_fields)
    print(f"{len(done)} runs already done")
    edmonds_timeout = cfg.get("edmonds_timeout_sec", 60)
    edmonds_repeats = max(1, int(cfg.get("edmonds_repeats", 3)))
    edmonds_max_n = cfg.get("edmonds_max_n")  # vertex count; None = no cap
    edmonds_disabled = set()
    for r in _load_rows(COMPLEXITY_CSV):
        if r["algorithm"] == "edmonds" and r["timed_out"] == "True":
            edmonds_disabled.add(r["family"])

    complexity_cfg = cfg["complexity"]
    tasks = []
    for c in complexity_cfg["F1_sparse"]["c_values"]:
        # Family name carries c (and density for F2) so that the resume
        # key (family, n_param, seed, ...) distinguishes them.
        family = f"F1_sparse_c{c}"
        for n in complexity_cfg["F1_sparse"]["sizes"]:
            for seed in complexity_cfg["F1_sparse"]["seeds"]:
                tasks.append((family, "F1_sparse", n, n, seed, (c,), f"c{c}"))
    for d in complexity_cfg["F2_dense"]["densities"]:
        family = f"F2_density{round(d * 100)}"
        for n in complexity_cfg["F2_dense"]["sizes"]:
            for seed in complexity_cfg["F2_dense"]["seeds"]:
                tasks.append(
                    (family, "F2_dense", n, n, seed, (d,), f"d{round(d * 100)}")
                )
    for n in complexity_cfg["F3_bad_greedy"]["sizes"]:
        for seed in complexity_cfg["F3_bad_greedy"]["seeds"]:
            tasks.append(("F3_bad_greedy", "F3_bad_greedy", n, n, seed, (), ""))
    for n in complexity_cfg["F4_nested_blossoms"]["sizes"]:
        for seed in complexity_cfg["F4_nested_blossoms"]["seeds"]:
            tasks.append(
                ("F4_nested_blossoms", "F4_nested_blossoms", n, n, seed, (), "")
            )
    for n_param in complexity_cfg["F5_chains"]["n_params"]:
        for seed in complexity_cfg["F5_chains"]["seeds"]:
            tasks.append(("F5_chains", "F5_chains", n_param, n_param, seed, (), ""))
    for n in complexity_cfg["F6_long_path"]["sizes"]:
        for seed in complexity_cfg["F6_long_path"]["seeds"]:
            tasks.append(("F6_long_path", "F6_long_path", n, n, seed, (), ""))

    pending = []
    for family, builder_key, n, n_param, seed, builder_args, _tag in tasks:
        variants = ["greedy_on", "greedy_off"] + (
            ["edgescan"] if builder_key == "F5_chains" else []
        )
        gkeys = [(family, str(n_param), str(seed), "gabow", v) for v in variants]
        ekey = (family, str(n_param), str(seed), "edmonds", "none")
        too_big = (
            edmonds_max_n is not None
            and _expected_vertices(builder_key, n) > edmonds_max_n
        )
        want_edmonds = (
            ekey not in done and family not in edmonds_disabled and not too_big
        )
        if all(k in done for k in gkeys) and not want_edmonds:
            continue
        pending.append(
            (
                family,
                builder_key,
                n,
                n_param,
                seed,
                builder_args,
                edmonds_timeout,
                want_edmonds,
                edmonds_repeats,
            )
        )

    print(f"{len(pending)} tasks to run")
    t0 = time.perf_counter()
    n_done = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(complexity_task, args): args for args in pending}
        for fut in as_completed(futures):
            family, _bk, n, n_param, seed = futures[fut][:5]
            try:
                rows = fut.result()
            except Exception as e:  # noqa: BLE001 -- one failed task must not abort the pool; it has no row, so a rerun retries it
                print(f"ERROR {family} n={n_param} seed={seed}: {e}")
                continue
            for row in rows:
                _append_row(COMPLEXITY_CSV, COMPLEXITY_FIELDS, row)
                if row["algorithm"] == "edmonds" and row["timed_out"] == True:
                    print(
                        f"  Edmonds timed out: {family} n_param={n_param} seed={seed}"
                    )
            n_done += 1
            if n_done % 5 == 0:
                print(
                    f"...{n_done}/{len(pending)} tasks, {time.perf_counter() - t0:.1f}s"
                )
    print(f"Complexity done: {time.perf_counter() - t0:.1f}s")


def report_complexity():
    rows = _load_rows(COMPLEXITY_CSV)
    if not rows:
        print("no complexity data yet")
        return
    print(
        f"{'family':20} {'n_param':10} {'variant':10} {'median_sec':12} {'iterations':10}"
    )
    keys = sorted(
        {(r["family"], r["n_param"]) for r in rows}, key=lambda t: (t[0], int(t[1]))
    )
    for family, n_param in keys:
        sub = [r for r in rows if r["family"] == family and r["n_param"] == n_param]
        for algo_variant in sorted({(r["algorithm"], r["variant"]) for r in sub}):
            algo, variant = algo_variant
            # Edmonds rows already hold the per-graph median of their repeats.
            vals = [
                float(r["runtime_sec"])
                for r in sub
                if r["algorithm"] == algo
                and r["variant"] == variant
                and r["timed_out"] != "True"
            ]
            label = "edmonds" if algo == "edmonds" else variant
            iters = next(
                (
                    r["iterations"]
                    for r in sub
                    if r["algorithm"] == algo and r["variant"] == variant
                ),
                "",
            )
            if vals:
                print(
                    f"{family:20} {n_param:10} {label:10} {statistics.median(vals):12.4g} {iters!s:10}"
                )
            else:
                print(
                    f"{family:20} {n_param:10} {label:10} {'TIMED OUT':12} {iters!s:10}"
                )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--report", action="store_true")
    p.add_argument("--config", default=str(CONFIG_PATH))
    args = p.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    max_workers = cfg.get("max_workers") or max(1, (os.cpu_count() or 2) - 1)

    if args.report:
        report_correctness()
        print()
        report_complexity()
        return

    info = env_info()
    print("Environment:", json.dumps(info, indent=2))
    (RESULTS / "environment.json").write_text(json.dumps(info, indent=2))

    t0 = time.perf_counter()
    run_correctness(cfg["correctness"], max_workers)
    run_complexity(cfg, max_workers)
    print(f"\nTotal run time: {time.perf_counter() - t0:.1f}s")
    print()
    report_correctness()
    print()
    report_complexity()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
