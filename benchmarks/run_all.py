"""Single entry point for Stage B (operation counts, Gabow only) and Stage C
(wall-clock, Gabow vs. NetworkX's Edmonds-based max_weight_matching), in
support of experimental (not proof) evidence for max_cardinality_matching_gabow
running in O(sqrt(n) * m * alpha(n)).

Reuses, rather than duplicates:
  * benchmarks/bench_matching.py  -- density helper, adaptive-repeats timing
    helper, non-mutation check.
  * benchmarks/verify_complexity.py -- the `_counters`-based `measure()`
    helper and the F3/F4/F5 graph generators (bad_greedy_graph,
    nested_blossoms_graph, chains_graph).
Does not modify either file, does not touch backups/, does not change any
algorithm logic -- every measurement goes through the existing, unmodified
`max_cardinality_matching_gabow(_counters=..., _skip_greedy_init=...)` and
`max_weight_matching(maxcardinality=True)`.

Usage:
    python benchmarks/run_all.py --quick     # small n, a few minutes
    python benchmarks/run_all.py --full       # full sweep (slow)

Outputs under benchmarks/results/:
    stageB_*.csv, stageC_*.csv   (raw data, one row per run)
    plots/P1_*.png ... plots/P10_*.png  (no P7: superseded by P10)
    SUMMARY.md
"""

import argparse
import csv
import functools
import math
import os
import platform
import random
import statistics
import sys
import time
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter

import networkx as nx

sys.path.insert(0, str(Path(__file__).parent))
import bench_matching as BM
import verify_complexity as VC

OUTDIR = Path(__file__).with_name("results")
PLOTDIR = OUTDIR / "plots"
OUTDIR.mkdir(parents=True, exist_ok=True)
PLOTDIR.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]  # >= 5 seeds per (family, n), as specified

# Markers distinguish families, so plots stay readable without color.
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


# ---------------------------------------------------------------------------
# Graph families (F1 F2 built here; F3-F6 reuse/extend verify_complexity.py)
# ---------------------------------------------------------------------------


def f1_random_sparse(n, c, seed):
    return nx.gnm_random_graph(n, c * n, seed=seed)


def f2_random_dense(n, density, seed):
    m = BM.edges_for_density(n, density)
    return nx.gnm_random_graph(n, m, seed=seed)


def f3_bad_greedy(n, seed):
    # bad_greedy_graph(k) has 4k nodes; seed is irrelevant (deterministic
    # construction) but accepted for a uniform call signature.
    k = max(1, n // 4)
    return VC.bad_greedy_graph(k)


def f4_nested_blossoms(n, seed):
    # nested_blossoms_graph(depth) has 2*depth + 1 nodes.
    depth = max(1, (n - 1) // 2)
    return VC.nested_blossoms_graph(depth)


def f5_short_and_long_chains(n_param, seed, shuffle=False):
    G = VC.chains_graph(n_param, mode=1)
    if shuffle:
        rng = random.Random(seed)
        nodes = list(G.nodes())
        relabel = dict(zip(nodes, rng.sample(nodes, len(nodes))))
        edges = [(relabel[u], relabel[v]) for u, v in G.edges()]
        rng.shuffle(edges)
        H = nx.Graph()
        H.add_nodes_from(relabel.values())
        H.add_edges_from(edges)
        return H
    return G


def f6_long_path(n, seed):
    return nx.path_graph(n)


def _f5_true_edge_order(n_param, mode=1):
    """The TRUE edge-insertion order for VC.chains_graph(n_param, mode) --
    NOT nx.Graph.edges(), which is node-major and does not preserve global
    insertion order (verified empirically: a late edge touching an early,
    high-degree node gets pulled forward in G.edges()'s own iteration).
    Mirrors verify_complexity.complete_g / _add_chain / chains_graph's EXACT
    id-assignment and edge-adding sequence (read-only -- does not change
    that file), needed to replicate the companion-page C++ reference's own
    edge-scan greedy init() faithfully. See GabowRevised.h lines 509-521.
    """
    order = []
    m_param = 4 * n_param
    n_clique = max(2, int(math.sqrt(m_param)))
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


def _edge_scan_greedy(edge_order):
    """Global edge-scan greedy matching: scan edges in insertion order,
    match (u, v) if both are still free. Exactly what the companion-page
    C++ reference's GabowRevised.h init() does (forall_edges(e,G): if
    u!=v and both mate==nil: match) -- see the investigation that added
    this (F5 initial-matching experiment).
    """
    mate = {}
    for u, v in edge_order:
        if u != v and u not in mate and v not in mate:
            mate[u] = v
            mate[v] = u
    return mate


# ---------------------------------------------------------------------------
# Stage B: operation counts (Gabow only)
# ---------------------------------------------------------------------------

STAGEB_FIELDS = [
    "family",
    "n",
    "n_param",
    "m",
    "seed",
    "greedy_init",
    "iterations",
    "delta_phases",
    "total_ops",
    "edge_scans",
    "search_steps",
    "blossom_contraction_events",
    "uf_find_calls",
    "augmentations",
]


def _measure_stageB(G, family, n_label, seed, greedy_init, _initial_mate=None):
    """n_label is the generator's own size parameter (e.g. n_param for the
    F5 chains family, which is NOT the resulting node count -- recorded as
    its own `n_param` column for traceability; `n` is always the actual
    G.number_of_nodes(), which is what every plot's x-axis uses).

    _initial_mate, if given, overrides greedy_init entirely (same rule as
    the real max_cardinality_matching_gabow): used for the F5 edge-scan-
    greedy start mode, where greedy_init's True/False distinction does not
    apply -- the column is still written (as False) for schema uniformity.
    """
    counters = {}
    nx.max_cardinality_matching_gabow(
        G,
        _counters=counters,
        _skip_greedy_init=not greedy_init,
        _initial_mate=_initial_mate,
    )
    total_ops = sum(counters[k] for k in VC.OP_KEYS)
    return {
        "family": family,
        "n": G.number_of_nodes(),
        "n_param": n_label,
        "m": G.number_of_edges(),
        "seed": seed,
        "greedy_init": greedy_init,
        "iterations": counters["iterations"],
        "delta_phases": counters["delta_phases"],
        "total_ops": total_ops,
        "edge_scans": counters["edge_scans"],
        "search_steps": counters["search_steps"],
        "blossom_contraction_events": counters["blossom_contraction_events"],
        "uf_find_calls": counters["uf_base_find_calls"]
        + counters["uf_dbase_find_calls"],
        "augmentations": counters["augmentations"],
    }


def _write_rows_csv(rows, name, fields):
    path = OUTDIR / name
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return path


def _append_row_csv(row, name, fields):
    """Append one finished row to disk immediately (resumability: a killed
    run loses at most the one in-flight row, not everything collected so
    far). Creates the file with a header on first use.
    """
    path = OUTDIR / name
    exists = path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow(row)
    return path


def _load_done_keys(name, key_fields):
    """Set of key-field tuples (as strings, matching what csv.DictReader
    hands back) already present in an existing results CSV -- so a
    restarted run can skip work it already did. Empty set if the file
    doesn't exist yet.
    """
    path = OUTDIR / name
    if not path.exists():
        return set()
    done = set()
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            done.add(tuple(r[k] for k in key_fields))
    return done


def _coerce_bool_or_none(s):
    if s == "True":
        return True
    if s == "False":
        return False
    return None


def _load_rows_csv(name, int_fields=(), float_fields=(), bool_fields=()):
    """Load a previously-written results CSV back into typed dicts (csv
    round-trips everything as strings). Used to regenerate plots/SUMMARY.md
    from already-collected data without re-running Stage B/C collection.
    """
    path = OUTDIR / name
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            row = dict(r)
            for k in int_fields:
                row[k] = int(row[k]) if row[k] != "" else None
            for k in float_fields:
                row[k] = float(row[k]) if row[k] != "" else None
            for k in bool_fields:
                row[k] = _coerce_bool_or_none(row[k])
            rows.append(row)
    return rows


def load_stageb_csv(name="stageB_raw.csv"):
    return _load_rows_csv(
        name,
        int_fields=[k for k in STAGEB_FIELDS if k not in ("family", "greedy_init")],
        bool_fields=["greedy_init"],
    )


def load_stagec_csv(name="stageC_raw.csv"):
    rows = _load_rows_csv(
        name,
        int_fields=["n", "m", "seed", "iterations"],
        float_fields=["runtime_sec"],
        bool_fields=["greedy_init", "greedy_already_optimal"],
    )
    return rows


STAGEB_KEY_FIELDS = ["family", "n_param", "seed", "greedy_init"]


def run_stage_b(sizes_by_family, n_seeds):
    print("\n" + "=" * 70)
    print("STAGE B: operation counts (Gabow only) [resumable]")
    print("=" * 70)
    seeds = SEEDS[:n_seeds]
    done = _load_done_keys("stageB_raw.csv", STAGEB_KEY_FIELDS)
    print(f"  {len(done)} rows already done, will be skipped")

    def add(family, gen, sizes, extra_kwargs=None):
        extra_kwargs = extra_kwargs or {}
        n_new = 0
        for n in sizes:
            for seed in seeds:
                G = None
                for greedy_init in (True, False):
                    key = (family, str(n), str(seed), str(greedy_init))
                    if key in done:
                        continue
                    if G is None:
                        G = gen(n, seed, **extra_kwargs)
                    row = _measure_stageB(G, family, n, seed, greedy_init)
                    _append_row_csv(row, "stageB_raw.csv", STAGEB_FIELDS)
                    done.add(key)
                    n_new += 1
        print(
            f"  {family}: {len(sizes)} sizes x {len(seeds)} seeds x 2 -- {n_new} new rows"
        )

    for c in (3, 5, 10):
        add(
            f"F1_sparse_c{c}",
            lambda n, seed, c=c: f1_random_sparse(n, c, seed),
            sizes_by_family["F1"],
        )
    for d in (0.10, 0.25, 0.50, 0.75, 1.00):
        add(
            f"F2_density{round(d * 100)}",
            lambda n, seed, d=d: f2_random_dense(n, d, seed),
            sizes_by_family["F2"],
        )
    add("F3_bad_greedy", f3_bad_greedy, sizes_by_family["F3"])
    add("F4_nested_blossoms", f4_nested_blossoms, sizes_by_family["F4"])

    n_new_f5 = 0
    for n_param in sizes_by_family["F5"]:
        for seed in seeds[:3]:  # "3 shuffles per size"
            for family_name, shuffle in (
                ("F5_chains_unshuffled", False),
                ("F5_chains_shuffled", True),
            ):
                key = (family_name, str(n_param), str(seed), "True")
                key2 = (family_name, str(n_param), str(seed), "False")
                if key in done and key2 in done:
                    continue
                G = f5_short_and_long_chains(n_param, seed, shuffle=shuffle)
                for greedy_init in (True, False):
                    k = (family_name, str(n_param), str(seed), str(greedy_init))
                    if k in done:
                        continue
                    row = _measure_stageB(G, family_name, n_param, seed, greedy_init)
                    _append_row_csv(row, "stageB_raw.csv", STAGEB_FIELDS)
                    done.add(k)
                    n_new_f5 += 1
    print(f"  F5_chains: {n_new_f5} new rows")

    add("F6_long_path", f6_long_path, sizes_by_family["F6"])

    all_rows = load_stageb_csv()
    print(f"Stage B raw data: {OUTDIR / 'stageB_raw.csv'} ({len(all_rows)} rows total)")

    families = {
        "F1": [r for r in all_rows if r["family"].startswith("F1")],
        "F2": [r for r in all_rows if r["family"].startswith("F2")],
        "F3": [r for r in all_rows if r["family"] == "F3_bad_greedy"],
        "F4": [r for r in all_rows if r["family"] == "F4_nested_blossoms"],
        "F5": [r for r in all_rows if r["family"].startswith("F5_chains_")],
        "F6": [r for r in all_rows if r["family"] == "F6_long_path"],
    }
    return all_rows, families


# ---------------------------------------------------------------------------
# Stage C: wall-clock, Gabow (greedy ON and OFF) vs Edmonds
# ---------------------------------------------------------------------------

# algorithm is "gabow" (greedy_init True/False) or "edmonds" (greedy_init
# always None -- Edmonds has no such concept). iterations/greedy_already_
# optimal are gabow-only (None for edmonds rows), obtained from one
# untimed diagnostic call per (family, n, seed, greedy_init) -- NOT from
# the timed calls themselves, so the timing measurements below remain free
# of instrumentation overhead, per the original Stage C spec.
STAGEC_FIELDS = [
    "family",
    "n",
    "m",
    "seed",
    "algorithm",
    "greedy_init",
    "runtime_sec",
    "iterations",
    "greedy_already_optimal",
]
TIMEOUT_SEC = 120.0
MIN_REPEATS = 5
TIME_BUDGET_SEC = 2.0


SLOW_CALL_REPEAT_THRESHOLD_SEC = 10.0


def _adaptive_timed_runs(fn, G):
    """>= MIN_REPEATS runs, or until TIME_BUDGET_SEC total elapsed, whichever
    is reached last (so cheap calls still get >= MIN_REPEATS samples) --
    EXCEPT once a single call is measured to take longer than
    SLOW_CALL_REPEAT_THRESHOLD_SEC (~10s), in which case we stop after that
    ONE call rather than forcing MIN_REPEATS: for slow calls, the intended
    statistical spread comes from the >= 5 different seeds each (family, n)
    point is run at, not from repeating the same graph 5 times at tens of
    seconds each. (An earlier version of this function did NOT have this
    early exit -- the `len(times) < MIN_REPEATS` half of the original loop
    condition forced 5 full repeats regardless of how slow each one was,
    which would have made F1's capped Edmonds comparison ~5x slower than
    estimated; caught before --full was run, not after.) Returns (times,
    result_of_last_call); `times` has length 1 for any slow call, so this
    is visible directly in stageC_raw.csv as well, not just in this
    function's behavior.
    """
    times = []
    result = None
    total = 0.0
    while len(times) < MIN_REPEATS or total < TIME_BUDGET_SEC:
        Gc = G.copy()
        t0 = time.perf_counter()
        result = fn(Gc)
        dt = time.perf_counter() - t0
        times.append(dt)
        total += dt
        if dt > SLOW_CALL_REPEAT_THRESHOLD_SEC:
            break
        if dt > TIMEOUT_SEC:
            break
        if len(times) >= 50:  # hard cap so a very fast call can't loop forever
            break
    return times, result


def _gabow_diagnostics(G, greedy_init):
    """One untimed call with _counters, separate from the timed runs, to
    get iterations and whether greedy alone already found the maximum
    matching (augmentations == 0 -- i.e. the only iteration run was the
    single "confirm no augmenting path exists" search). Deterministic
    given (G, greedy_init), so one call suffices rather than one per repeat.
    """
    counters = {}
    nx.max_cardinality_matching_gabow(
        G.copy(), _counters=counters, _skip_greedy_init=not greedy_init
    )
    return counters["iterations"], counters["augmentations"] == 0


def _calibrate_f1_edmonds_cap(sizes_by_family):
    """F1 includes n up to 10**5 for Gabow, but Edmonds (O(n**3)) cannot
    follow it there. Time ONE real Edmonds run at F1 n=20000, c=3 (if 20000
    or larger is actually requested); if it takes more than ~120s, cap
    Edmonds (ONLY Edmonds -- Gabow still runs at every F1 size) at 5000
    instead of 20000. Returns the chosen cap (or None if n=20000 was never
    going to be attempted anyway, meaning no cap is needed).
    """
    f1_sizes = sizes_by_family.get("F1", [])
    if not any(n >= 20000 for n in f1_sizes):
        return None
    print("\nCalibrating Edmonds F1 cap: timing ONE Edmonds run at n=20000, c=3 ...")
    G = f1_random_sparse(20000, 3, 0)
    t0 = time.perf_counter()
    nx.max_weight_matching(G, maxcardinality=True)
    dt = time.perf_counter() - t0
    cap = 20000 if dt <= 120.0 else 5000
    print(
        f"  measured {dt:.1f}s at n=20000 -> Edmonds cap for F1 set to {cap} "
        f"(Gabow is NOT capped and still runs at every requested F1 size, including 10**5)"
    )
    return cap


def _reconstruct_edmonds_disabled(csv_name, timeout_sec):
    """On resume, re-derive which families had Edmonds already shown too
    slow (so a restart doesn't retry Edmonds on a large n that a prior,
    killed run already proved too slow for -- it would just time out
    again)."""
    path = OUTDIR / csv_name
    disabled = {}
    if not path.exists():
        return disabled
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["algorithm"] == "edmonds" and float(r["runtime_sec"]) > timeout_sec:
                disabled[r["family"]] = True
    return disabled


def run_stage_c(sizes_by_family, n_seeds, csv_name="stageC_raw.csv"):
    print("\n" + "=" * 70)
    print(
        f"STAGE C: wall-clock, Gabow (greedy ON/OFF) vs Edmonds (instrumentation OFF for timing) [resumable -> {csv_name}]"
    )
    print("=" * 70)
    seeds = SEEDS[:n_seeds]
    # Keyed by the FULL family string (e.g. "F1_sparse_c3"), NOT by fam_key
    # ("F1") -- an earlier version shared one flag across all three F1
    # c-values (and all five F2 densities), so one slow Edmonds call in
    # F1_sparse_c3 silently disabled Edmonds for F1_sparse_c5/c10 too, even
    # though they were never themselves tested. fam_key is still used below
    # for the (legitimately shared) F1 size cap.
    edmonds_disabled = _reconstruct_edmonds_disabled(csv_name, TIMEOUT_SEC)
    f1_edmonds_cap = _calibrate_f1_edmonds_cap(sizes_by_family)
    # Combo-level done-set: (family, n, seed, algorithm, greedy_init) with
    # no runtime_sec -- a whole combo's repeats are written atomically, so
    # presence of ANY row for a combo means that combo is done.
    done = set()
    path = OUTDIR / csv_name
    if path.exists():
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                done.add(
                    (r["family"], r["n"], r["seed"], r["algorithm"], r["greedy_init"])
                )
    print(f"  {len(done)} (family,n,seed,algorithm,greedy_init) combos already done")

    def gabow(greedy_init):
        return lambda G: nx.max_cardinality_matching_gabow(
            G, _skip_greedy_init=not greedy_init
        )

    def edmonds(G):
        return nx.max_weight_matching(G, maxcardinality=True)

    def run_family(family, gen, sizes, extra_kwargs=None):
        extra_kwargs = extra_kwargs or {}
        for n in sizes:
            for seed in seeds:
                g_combo_keys = {
                    gi: (family, str(n), str(seed), "gabow", str(gi))
                    for gi in (True, False)
                }
                e_key = (family, str(n), str(seed), "edmonds", "None")
                need_gabow = any(k not in done for k in g_combo_keys.values())
                fam_key = family.split("_")[0]
                over_f1_cap = (
                    fam_key == "F1"
                    and f1_edmonds_cap is not None
                    and n > f1_edmonds_cap
                )
                need_edmonds = (
                    e_key not in done
                    and not edmonds_disabled.get(family, False)
                    and not over_f1_cap
                )
                if not need_gabow and not need_edmonds:
                    continue

                G = gen(n, seed, **extra_kwargs)
                m = G.number_of_edges()

                res_g_by_mode = {}
                for greedy_init in (True, False):
                    key = g_combo_keys[greedy_init]
                    iters, already_opt = _gabow_diagnostics(G, greedy_init)
                    if key in done:
                        # Still need res_g for the Edmonds size-match assert
                        # below if Edmonds itself is new this run.
                        if need_edmonds:
                            res_g_by_mode[greedy_init] = gabow(greedy_init)(G)
                        continue
                    times_g, res_g = _adaptive_timed_runs(gabow(greedy_init), G)
                    res_g_by_mode[greedy_init] = res_g
                    for t in times_g:
                        _append_row_csv(
                            {
                                "family": family,
                                "n": n,
                                "m": m,
                                "seed": seed,
                                "algorithm": "gabow",
                                "greedy_init": greedy_init,
                                "runtime_sec": t,
                                "iterations": iters,
                                "greedy_already_optimal": already_opt,
                            },
                            csv_name,
                            STAGEC_FIELDS,
                        )
                    done.add(key)
                    print(
                        f"  {family} n={n} seed={seed} greedy_init={greedy_init}: "
                        f"median={statistics.median(times_g):.4g}s mean={statistics.mean(times_g):.4g}s "
                        f"iterations={iters} greedy_already_optimal={already_opt}"
                    )

                if over_f1_cap and not edmonds_disabled.get(family, False):
                    print(
                        f"    skipping Edmonds at n={n} ({family}): over the calibrated F1 cap "
                        f"({f1_edmonds_cap}); Gabow still ran above."
                    )
                if need_edmonds:
                    times_e, res_e = _adaptive_timed_runs(edmonds, G)
                    if max(times_e) > TIMEOUT_SEC:
                        print(
                            f"    Edmonds exceeded {TIMEOUT_SEC}s at n={n} ({family}); "
                            "disabling Edmonds for larger n in this family, continuing Gabow."
                        )
                        edmonds_disabled[family] = True
                    for t in times_e:
                        _append_row_csv(
                            {
                                "family": family,
                                "n": n,
                                "m": m,
                                "seed": seed,
                                "algorithm": "edmonds",
                                "greedy_init": None,
                                "runtime_sec": t,
                                "iterations": None,
                                "greedy_already_optimal": None,
                            },
                            csv_name,
                            STAGEC_FIELDS,
                        )
                    done.add(e_key)
                    for greedy_init, res_g in res_g_by_mode.items():
                        if (
                            len(res_g) != len(res_e)
                            or not nx.is_matching(G, res_g)
                            or not nx.is_matching(G, res_e)
                        ):
                            raise AssertionError(
                                f"MISMATCH at family={family} n={n} seed={seed} "
                                f"greedy_init={greedy_init}: gabow size={len(res_g)} "
                                f"edmonds size={len(res_e)} -- aborting, this should never happen."
                            )
                    print(
                        f"    edmonds: median={statistics.median(times_e):.4g}s mean={statistics.mean(times_e):.4g}s"
                    )

    for c in (3, 5, 10):
        run_family(
            f"F1_sparse_c{c}",
            lambda n, seed, c=c: f1_random_sparse(n, c, seed),
            sizes_by_family["F1"],
        )
    for d in (0.10, 0.25, 0.50, 0.75, 1.00):
        run_family(
            f"F2_density{round(d * 100)}",
            lambda n, seed, d=d: f2_random_dense(n, d, seed),
            sizes_by_family["F2"],
        )

    rows = load_stagec_csv(csv_name)
    print(f"Stage C raw data: {OUTDIR / csv_name} ({len(rows)} rows total)")
    return rows


# ---------------------------------------------------------------------------
# Hard families (F3-F6) vs Edmonds: P8/P9. Separate CSV and separate
# (lower) Edmonds-disable threshold from F1/F2's run_stage_c, since these
# families are specifically adversarial for Edmonds and the point is
# comparative speed, not a from-scratch repeat of Stage C's own policy.
# ---------------------------------------------------------------------------

STAGEC_HARD_FIELDS = [
    "family",
    "n",
    "n_param",
    "m",
    "seed",
    "algorithm",
    "variant",
    "runtime_sec",
    "iterations",
]
HARD_EDMONDS_TIMEOUT_SEC = 60.0


def run_stage_c_hard(
    sizes_by_family, n_seeds, edmonds_timeout=HARD_EDMONDS_TIMEOUT_SEC
):
    """F3-F6 vs Edmonds: Gabow greedy ON, greedy OFF, and (F5 only) the
    edge-scan-greedy start, each vs Edmonds capped at `edmonds_timeout`.
    Resumable: appends to stageC_hardfamilies.csv, skips combos already
    present.

    `n` is the actual vertex count G.number_of_nodes() (what P8/P9 plot);
    `n_param` is the generator's own size input (the resume key, since
    `n` is only known after building the graph).
    """
    csv_name = "stageC_hardfamilies.csv"
    print("\n" + "=" * 70)
    print(f"STAGE C (hard families F3-F6 vs Edmonds) [resumable -> {csv_name}]")
    print("=" * 70)
    seeds = SEEDS[:n_seeds]
    edmonds_disabled = _reconstruct_edmonds_disabled_hard(csv_name, edmonds_timeout)
    done = set()
    path = OUTDIR / csv_name
    if path.exists():
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                done.add(
                    (r["family"], r["n_param"], r["seed"], r["algorithm"], r["variant"])
                )
    print(f"  {len(done)} combos already done")

    def gabow_variant(G, variant, n_param):
        if variant == "greedy_on":
            counters = {}
            m = nx.max_cardinality_matching_gabow(
                G.copy(), _counters=counters, _skip_greedy_init=False
            )
            return m, counters["iterations"]
        if variant == "greedy_off":
            counters = {}
            m = nx.max_cardinality_matching_gabow(
                G.copy(), _counters=counters, _skip_greedy_init=True
            )
            return m, counters["iterations"]
        if variant == "edgescan":
            # The C++ reference's TRUE edge-insertion order, exactly as in
            # run_f5_edgescan (P10) -- NOT G.edges(), whose node-major order
            # yields a different greedy matching that pre-solves this
            # family in one iteration.
            edge_order = _f5_true_edge_order(n_param, mode=1)
            assert {frozenset(e) for e in edge_order} == {
                frozenset(e) for e in G.edges()
            }
            initial_mate = _edge_scan_greedy(edge_order)
            counters = {}
            m = nx.max_cardinality_matching_gabow(
                G.copy(),
                _counters=counters,
                _skip_greedy_init=True,
                _initial_mate=initial_mate,
            )
            return m, counters["iterations"]
        raise ValueError(variant)

    def edmonds(G):
        return nx.max_weight_matching(G, maxcardinality=True)

    def run_family(family, gen, sizes, variants, extra_kwargs=None):
        extra_kwargs = extra_kwargs or {}
        for n in sizes:
            for seed in seeds:
                variant_keys = {
                    v: (family, str(n), str(seed), "gabow", v) for v in variants
                }
                e_key = (family, str(n), str(seed), "edmonds", "none")
                need_any_gabow = any(k not in done for k in variant_keys.values())
                need_edmonds = e_key not in done and not edmonds_disabled.get(
                    family, False
                )
                if not need_any_gabow and not need_edmonds:
                    continue
                G = gen(n, seed, **extra_kwargs)
                n_real = G.number_of_nodes()
                m = G.number_of_edges()
                res_by_variant = {}
                for variant in variants:
                    key = variant_keys[variant]
                    if key in done:
                        if need_edmonds:
                            res_by_variant[variant], _ = gabow_variant(G, variant, n)
                        continue
                    t0 = time.perf_counter()
                    res, iters = gabow_variant(G, variant, n)
                    dt = time.perf_counter() - t0
                    res_by_variant[variant] = res
                    _append_row_csv(
                        {
                            "family": family,
                            "n": n_real,
                            "n_param": n,
                            "m": m,
                            "seed": seed,
                            "algorithm": "gabow",
                            "variant": variant,
                            "runtime_sec": dt,
                            "iterations": iters,
                        },
                        csv_name,
                        STAGEC_HARD_FIELDS,
                    )
                    done.add(key)
                    print(
                        f"  {family} n_param={n} n={n_real} seed={seed} {variant}: {dt:.4g}s iterations={iters}"
                    )
                sizes_seen = {len(r) for r in res_by_variant.values()}
                if len(sizes_seen) > 1:
                    raise AssertionError(
                        f"MISMATCH between Gabow variants family={family} n_param={n} seed={seed}: "
                        f"{ {v: len(r) for v, r in res_by_variant.items()} }"
                    )
                if need_edmonds:
                    t0 = time.perf_counter()
                    res_e = edmonds(G)
                    dt = time.perf_counter() - t0
                    _append_row_csv(
                        {
                            "family": family,
                            "n": n_real,
                            "n_param": n,
                            "m": m,
                            "seed": seed,
                            "algorithm": "edmonds",
                            "variant": "none",
                            "runtime_sec": dt,
                            "iterations": None,
                        },
                        csv_name,
                        STAGEC_HARD_FIELDS,
                    )
                    done.add(e_key)
                    for variant, res_g in res_by_variant.items():
                        if len(res_g) != len(res_e):
                            raise AssertionError(
                                f"MISMATCH family={family} n={n} seed={seed} variant={variant}: "
                                f"gabow={len(res_g)} edmonds={len(res_e)}"
                            )
                    print(f"    edmonds: {dt:.4g}s")
                    if dt > edmonds_timeout:
                        print(
                            f"    Edmonds exceeded {edmonds_timeout}s ({family}); disabling for larger n."
                        )
                        edmonds_disabled[family] = True

    run_family(
        "F3_bad_greedy",
        f3_bad_greedy,
        sizes_by_family["F3"],
        ("greedy_on", "greedy_off"),
    )
    run_family(
        "F4_nested_blossoms",
        f4_nested_blossoms,
        sizes_by_family["F4"],
        ("greedy_on", "greedy_off"),
    )
    run_family(
        "F5_chains_edgescan_hard",
        lambda n, seed: f5_short_and_long_chains(n, seed, shuffle=False),
        sizes_by_family["F5"],
        ("greedy_on", "greedy_off", "edgescan"),
    )
    run_family(
        "F6_long_path", f6_long_path, sizes_by_family["F6"], ("greedy_on", "greedy_off")
    )

    rows = _load_rows_csv(
        csv_name,
        int_fields=["n", "n_param", "m", "seed", "iterations"],
        float_fields=["runtime_sec"],
    )
    print(f"Stage C hard-families raw data: {path} ({len(rows)} rows total)")
    return rows


def _reconstruct_edmonds_disabled_hard(csv_name, timeout_sec):
    path = OUTDIR / csv_name
    disabled = {}
    if not path.exists():
        return disabled
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["algorithm"] == "edmonds" and float(r["runtime_sec"]) > timeout_sec:
                disabled[r["family"]] = True
    return disabled


# ---------------------------------------------------------------------------
# Stable F1 Edmonds check: the --full run's n=20000 Edmonds points are each
# a SINGLE sample (the slow-call repetition policy stops after one call past
# 10s), and one of them (258.5s) disagreed with an earlier calibration call
# on the identical graph/seed (69.3s) by 3.7x -- see "69s calibration vs
# ~260s measured" in SUMMARY.md. This runs a smaller, cheaper n (10000, where
# Edmonds is fast enough to afford real repetition) with REAL repeats for
# both algorithms, intended to be run in an isolated, otherwise-idle process
# so the numbers are not contending with anything else on the machine.
# ---------------------------------------------------------------------------

STABLE_CHECK_FIELDS = ["n", "c", "seed", "algorithm", "greedy_init", "runtime_sec"]
STABLE_CHECK_N = 10000
STABLE_CHECK_CS = (3, 5, 10)
STABLE_CHECK_SEEDS = (0, 1, 2)
STABLE_CHECK_EDMONDS_REPEATS = 3
STABLE_CHECK_GABOW_REPEATS = 5


def run_f1_stable_check(n=STABLE_CHECK_N, cs=STABLE_CHECK_CS, seeds=STABLE_CHECK_SEEDS):
    print("\n" + "=" * 70)
    print(
        f"STABLE F1 EDMONDS CHECK: n={n}, c in {cs}, seeds {seeds} "
        f"(edmonds x{STABLE_CHECK_EDMONDS_REPEATS}, gabow x{STABLE_CHECK_GABOW_REPEATS})"
    )
    print("=" * 70)

    def gabow_fn(greedy_init):
        return lambda G: nx.max_cardinality_matching_gabow(
            G, _skip_greedy_init=not greedy_init
        )

    def edmonds_fn(G):
        return nx.max_weight_matching(G, maxcardinality=True)

    rows = []
    for c in cs:
        for seed in seeds:
            G = f1_random_sparse(n, c, seed)

            times_e = []
            res_e = None
            for _ in range(STABLE_CHECK_EDMONDS_REPEATS):
                t0 = time.perf_counter()
                res_e = edmonds_fn(G.copy())
                times_e.append(time.perf_counter() - t0)
            for t in times_e:
                rows.append(
                    {
                        "n": n,
                        "c": c,
                        "seed": seed,
                        "algorithm": "edmonds",
                        "greedy_init": None,
                        "runtime_sec": t,
                    }
                )

            gabow_meds = {}
            for greedy_init in (True, False):
                times_g = []
                res_g = None
                for _ in range(STABLE_CHECK_GABOW_REPEATS):
                    t0 = time.perf_counter()
                    res_g = gabow_fn(greedy_init)(G.copy())
                    times_g.append(time.perf_counter() - t0)
                if (
                    len(res_g) != len(res_e)
                    or not nx.is_matching(G, res_g)
                    or not nx.is_matching(G, res_e)
                ):
                    raise AssertionError(
                        f"MISMATCH at n={n} c={c} seed={seed} greedy_init={greedy_init}: "
                        f"gabow size={len(res_g)} edmonds size={len(res_e)} -- aborting."
                    )
                for t in times_g:
                    rows.append(
                        {
                            "n": n,
                            "c": c,
                            "seed": seed,
                            "algorithm": "gabow",
                            "greedy_init": greedy_init,
                            "runtime_sec": t,
                        }
                    )
                gabow_meds[greedy_init] = statistics.median(times_g)

            print(
                f"  c={c} seed={seed}: edmonds median={statistics.median(times_e):.4g}s "
                f"min={min(times_e):.4g}s max={max(times_e):.4g}s | "
                f"gabow greedy_ON median={gabow_meds[True]:.4g}s | "
                f"gabow greedy_OFF median={gabow_meds[False]:.4g}s | "
                f"speedup(edmonds/gabow_OFF)={statistics.median(times_e) / gabow_meds[False]:.2f}x"
            )

    path = _write_rows_csv(rows, "stageC_stable_check.csv", STABLE_CHECK_FIELDS)
    print(f"Saved stable-check raw data to {path} ({len(rows)} rows total)")
    return rows


def load_stable_check_csv(name="stageC_stable_check.csv"):
    return _load_rows_csv(
        name,
        int_fields=["n", "c", "seed"],
        float_fields=["runtime_sec"],
        bool_fields=["greedy_init"],
    )


def _stable_check_table(stable_rows):
    """Per (c, seed): Edmonds, Gabow greedy-OFF and greedy-ON medians and the
    two speedups (Edmonds median / Gabow median), from an isolated
    run_f1_stable_check() collection; then the overall range of the
    greedy-OFF speedup.
    """
    if not stable_rows:
        return []
    out = [
        "| c | seed | Edmonds median (s) | Gabow OFF median (s) | Gabow ON median (s) | speedup OFF | speedup ON |",
        "|---|---|---|---|---|---|---|",
    ]
    speedups_off, speedups_on = [], []
    for c in sorted({r["c"] for r in stable_rows}):
        for seed in sorted({r["seed"] for r in stable_rows if r["c"] == c}):
            sub = [r for r in stable_rows if r["c"] == c and r["seed"] == seed]
            e = [r["runtime_sec"] for r in sub if r["algorithm"] == "edmonds"]
            g_on = [
                r["runtime_sec"]
                for r in sub
                if r["algorithm"] == "gabow" and r["greedy_init"] is True
            ]
            g_off = [
                r["runtime_sec"]
                for r in sub
                if r["algorithm"] == "gabow" and r["greedy_init"] is False
            ]
            if not (e and g_on and g_off):
                continue
            me, moff, mon = (
                statistics.median(e),
                statistics.median(g_off),
                statistics.median(g_on),
            )
            speedups_off.append(me / moff)
            speedups_on.append(me / mon)
            out.append(
                f"| {c} | {seed} | {me:.4g} | {moff:.4g} | {mon:.4g} | "
                f"{me / moff:.1f}x | {me / mon:.1f}x |"
            )
    if speedups_off:
        out.append(
            f"\nOverall range: greedy-OFF speedup {min(speedups_off):.1f}x-"
            f"{max(speedups_off):.1f}x, greedy-ON speedup "
            f"{min(speedups_on):.1f}x-{max(speedups_on):.1f}x (each Edmonds "
            f"median over {STABLE_CHECK_EDMONDS_REPEATS} runs, each Gabow median "
            f"over {STABLE_CHECK_GABOW_REPEATS} runs). The same check on the "
            "earlier code (LIFO bucket queue, before the FIFO fix) gave a "
            "greedy-OFF speedup of about 12x-44x (docs/HANDOFF.md).\n"
        )
    return out


def _fmt_x(v):
    return f"{v:.3g}x"


def _fmt_speedup(sp):
    if sp is None:
        return "--"
    med, lo, hi, k = sp
    if k == 1:
        return _fmt_x(med)
    return f"{_fmt_x(med)} [{lo:.3g}-{hi:.3g}]"


def _speedup_table(stagec_rows):
    """P6/P6b points (F1/F2): per-graph speedups, median [min-max]."""
    out = [
        "| family | n | greedy ON: speedup [per-graph min-max] | greedy OFF: speedup [min-max] | graphs | bimodal |",
        "|---|---|---|---|---|---|",
    ]
    on = _stagec_speedup_series(stagec_rows, "greedy_on")
    off = _stagec_speedup_series(stagec_rows, "greedy_off")
    for fam in _sorted_families(set(on) | set(off)):
        on_d, off_d = dict(on.get(fam, [])), dict(off.get(fam, []))
        for n in sorted(set(on_d) | set(off_d)):
            sp_on, sp_off = on_d.get(n), off_d.get(n)
            k = (sp_on or sp_off)[3]
            flags = [
                name for name, sp in (("ON", sp_on), ("OFF", sp_off)) if _is_bimodal(sp)
            ]
            out.append(
                f"| {fam} | {n} | {_fmt_speedup(sp_on)} | {_fmt_speedup(sp_off)} "
                f"| {k} | {', '.join(flags)} |"
            )
    return out


# ---------------------------------------------------------------------------
# F5 edge-scan-greedy start mode: an investigation found that the companion-
# page C++ reference (GabowRevised.h's init(), not our own vertex-scan
# greedy) is what actually produced the paper's Table 1 numbers, and that it
# leaves exactly the "chain endpoints exposed" starting state the paper's
# O(sqrt(n))-iterations argument assumes. Uses the real
# max_cardinality_matching_gabow's _initial_mate hook (added for this
# purpose), not a scratch copy -- unlike the earlier investigation.
# ---------------------------------------------------------------------------

F5_EDGESCAN_SIZES = (2500, 5000, 10000, 20000, 40000)


def run_f5_edgescan(sizes=F5_EDGESCAN_SIZES):
    """Deterministic (no seed dependence: unshuffled F5, edge-scan greedy
    has no randomness) -- one row per size. Each row's `n_param` is the
    generator's own size input; `n`/`m` are the actual built graph's
    totals.
    """
    print("\n" + "=" * 70)
    print(f"F5 EDGE-SCAN-GREEDY START: n_param in {sizes}")
    print("=" * 70)
    rows = []
    for n_param in sizes:
        G = VC.chains_graph(n_param, mode=1)
        edge_order = _f5_true_edge_order(n_param, mode=1)
        true_edges = {frozenset(e) for e in edge_order}
        real_edges = {frozenset(e) for e in G.edges()}
        assert true_edges == real_edges, (
            f"edge-order reconstruction mismatch at n_param={n_param} "
            "-- _f5_true_edge_order no longer matches verify_complexity.chains_graph"
        )
        initial_mate = _edge_scan_greedy(edge_order)
        row = _measure_stageB(
            G,
            "F5_chains_edgescan",
            n_param,
            seed=0,
            greedy_init=False,
            _initial_mate=initial_mate,
        )
        rows.append(row)
        print(
            f"  n_param={n_param:6d} n={row['n']:7d} m={row['m']:7d} "
            f"iterations={row['iterations']:4d} total_ops={row['total_ops']:10d} "
            f"ops/(sqrt(n)m)={row['total_ops'] / (math.sqrt(row['n']) * row['m']):.4f}"
        )
    path = _write_rows_csv(rows, "stageB_f5_edgescan.csv", STAGEB_FIELDS)
    print(f"Saved F5 edge-scan-greedy data to {path} ({len(rows)} rows total)")
    return rows


@functools.cache
def _simulate_heur_trigger_series(sizes=(5000, 10000, 20000)):
    """For each F5 edge-scan n_param, simulate GabowBeautified.h's `heur`
    trigger condition (number_of_iterations > 0.5*(max_size_of_M -
    size_of_M), checked BEFORE each iteration, using size_of_M as of the
    end of the previous one) against our own real per-iteration
    augmentation counts, and return (actual_n, trigger_iteration) pairs.
    Explains the companion driver's reported 24/33/47 (it stops counting
    once heur triggers, since heur then finishes everything in one
    further, uncounted sweep) -- see docs/complexity_audit.md.
    """
    out_n, out_trigger = [], []
    for n_param in sizes:
        G = VC.chains_graph(n_param, mode=1)
        edge_order = _f5_true_edge_order(n_param, mode=1)
        initial_mate = _edge_scan_greedy(edge_order)
        size_of_M = len(initial_mate) // 2
        n = G.number_of_nodes()
        max_size_of_M = min(n // 2, 2 * size_of_M)
        counters = {}
        nx.max_cardinality_matching_gabow(
            G, _counters=counters, _skip_greedy_init=True, _initial_mate=initial_mate
        )
        trigger = None
        for k, entry in enumerate(counters["per_iteration"], start=1):
            if k > 0.5 * (max_size_of_M - size_of_M):
                trigger = k
                break
            size_of_M += entry["augmentations"]
        if trigger is not None:
            out_n.append(n)
            out_trigger.append(trigger)
    return out_n, out_trigger


def load_f5_edgescan_csv(name="stageB_f5_edgescan.csv"):
    if not (OUTDIR / name).exists():
        return []
    return load_stageb_csv(name)


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------


def _median_iqr(values):
    values = sorted(values)
    med = statistics.median(values)
    if len(values) >= 4:
        q1 = statistics.median(values[: len(values) // 2])
        q3 = statistics.median(values[(len(values) + 1) // 2 :])
    else:
        q1 = q3 = med
    return med, q3 - q1


def _fit_loglog(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if y > 0]
    if len(pairs) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    xs2, ys2 = zip(*pairs)
    return VC._fit_loglog(xs2, ys2)


# ---------------------------------------------------------------------------
# Aggregation and plot style shared by all plots and SUMMARY.md
# ---------------------------------------------------------------------------

#: Families whose generator ignores the seed: at a given n every seed builds
#: the same graph, so all their timed runs at that n belong to one graph.
DETERMINISTIC_FAMILIES = frozenset(
    {"F3_bad_greedy", "F4_nested_blossoms", "F5_chains_edgescan_hard", "F6_long_path"}
)

#: A point whose per-graph speedups span more than this factor (max / min)
#: is treated as bimodal; the plots draw its per-graph min-max range.
BIMODAL_SPREAD = 3.0


def _graph_key(r):
    return 0 if r["family"] in DETERMINISTIC_FAMILIES else r["seed"]


def _per_graph_medians(rows):
    """{graph: median runtime over that graph's timed runs}."""
    by_graph = {}
    for r in rows:
        by_graph.setdefault(_graph_key(r), []).append(r["runtime_sec"])
    return {k: statistics.median(v) for k, v in by_graph.items()}


def _point_time(rows):
    """Wall-clock time of one (family, n, algorithm) point: the median over
    graphs of each graph's median over its timed runs. Fast calls are
    repeated until a time budget elapses, so a plain median over all runs
    would over-weight the easy graphs.
    """
    per_graph = _per_graph_medians(rows)
    return statistics.median(per_graph.values()) if per_graph else None


def _point_speedup(edmonds_rows, gabow_rows):
    """Speedup of one point from per-graph speedups (Edmonds median / Gabow
    median on the same graph, for the graphs timed with both). Returns
    (median, min, max, number of graphs), or None.
    """
    pe = _per_graph_medians(edmonds_rows)
    pg = _per_graph_medians(gabow_rows)
    common = sorted(set(pe) & set(pg))
    if not common:
        return None
    s = [pe[k] / pg[k] for k in common]
    return statistics.median(s), min(s), max(s), len(s)


def _is_bimodal(sp):
    return sp is not None and sp[3] > 1 and sp[2] / sp[1] > BIMODAL_SPREAD


#: One color, line style, marker and label per variant, used wherever
#: variants are compared (P4, P8, P10).
VARIANT_STYLE = {
    "greedy_on": ("tab:blue", "-", "o", "Gabow, greedy ON"),
    "greedy_off": ("tab:orange", "--", "s", "Gabow, greedy OFF"),
    "edgescan": ("tab:green", "-.", "^", "Gabow, edge-scan start (C++ init)"),
    "edmonds": ("tab:red", ":", "D", "Edmonds (max_weight_matching)"),
}

#: One color and marker per graph family, the same in every plot.
FAMILY_ORDER = [
    "F1_sparse_c3",
    "F1_sparse_c5",
    "F1_sparse_c10",
    "F2_density10",
    "F2_density25",
    "F2_density50",
    "F2_density75",
    "F2_density100",
    "F3_bad_greedy",
    "F4_nested_blossoms",
    "F5_chains_unshuffled",
    "F5_chains_shuffled",
    "F5_chains_edgescan",
    "F5_chains_edgescan_hard",
    "F6_long_path",
]
FAMILY_LABEL = {
    "F1_sparse_c3": "F1 random sparse, m = 3n",
    "F1_sparse_c5": "F1 random sparse, m = 5n",
    "F1_sparse_c10": "F1 random sparse, m = 10n",
    "F2_density10": "F2 random dense, 10%",
    "F2_density25": "F2 random dense, 25%",
    "F2_density50": "F2 random dense, 50%",
    "F2_density75": "F2 random dense, 75%",
    "F2_density100": "F2 complete graph",
    "F3_bad_greedy": "F3 bad-greedy gadgets",
    "F4_nested_blossoms": "F4 nested blossoms",
    "F5_chains_unshuffled": "F5 chains",
    "F5_chains_shuffled": "F5 chains, shuffled labels",
    "F5_chains_edgescan": "F5 chains, edge-scan start",
    "F5_chains_edgescan_hard": "F5 chains",
    "F6_long_path": "F6 long path",
}
_FAMILY_COLOR_INDEX = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3, 5, 7, 9, 11, 13, 15]


def _family_style(fam):
    """(color, marker) for a family, stable across plots."""
    i = FAMILY_ORDER.index(fam) if fam in FAMILY_ORDER else len(FAMILY_ORDER)
    colors = plt.get_cmap("tab20").colors
    return colors[_FAMILY_COLOR_INDEX[i % len(_FAMILY_COLOR_INDEX)]], MARKERS[
        i % len(MARKERS)
    ]


def _family_label(fam):
    return FAMILY_LABEL.get(fam, fam)


def _sorted_families(families):
    return sorted(
        families,
        key=lambda f: (FAMILY_ORDER.index(f) if f in FAMILY_ORDER else 99, f),
    )


def _legend_outside(ax, handles=None, **kw):
    """Legend to the right of the axes, so it never covers data."""
    kw.setdefault("fontsize", 8)
    if handles is not None:
        kw["handles"] = handles
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, **kw)


def _save(fig, outpath):
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _tidy_log_x(ax, ns):
    """On a log x axis spanning less than a factor of 30 (no or few powers
    of ten), label the measured n values instead of crowded minor ticks."""
    ns = sorted(set(ns))
    if ns and ns[-1] / ns[0] < 30:
        ax.set_xticks(ns)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:,.0f}"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def _bars_proxy():
    return Line2D(
        [],
        [],
        color="gray",
        marker="|",
        linestyle="none",
        markersize=12,
        label=f"per-graph min-max (bimodal: spread > {BIMODAL_SPREAD:g}x)",
    )


# ---------------------------------------------------------------------------
# Plots P1-P6, P8-P10 (P7 was removed: P10 supersedes it)
# ---------------------------------------------------------------------------


def _plot_stageb_metric(stageb_rows, outpath, metric, ylabel, title, reference=False):
    """P1-P3: one line per family, Gabow from the empty matching (stage B
    op counts), median over graphs at each n."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    off = [r for r in stageb_rows if r["greedy_init"] is False]
    for fam in _sorted_families({r["family"] for r in off}):
        sub = [r for r in off if r["family"] == fam]
        ns = sorted({r["n"] for r in sub})
        ys = []
        for n in ns:
            vals = [metric(r) for r in sub if r["n"] == n]
            vals = [v for v in vals if v is not None]
            ys.append(statistics.median(vals) if vals else float("nan"))
        color, marker = _family_style(fam)
        ax.plot(
            ns, ys, marker=marker, color=color, label=_family_label(fam), markersize=5
        )
    if reference:
        # c*sqrt(n), anchored at the SMALLEST measured n: a family that keeps
        # pace with sqrt(n) follows the line, a flat family falls below it.
        ns_all = sorted({r["n"] for r in stageb_rows})
        if ns_all:
            anchor_n = ns_all[0]
            anchor_val = max(
                1.0,
                statistics.median(
                    [r["iterations"] for r in stageb_rows if r["n"] == anchor_n]
                )
                or 1,
            )
            c = anchor_val / math.sqrt(anchor_n)
            ax.plot(
                ns_all,
                [c * math.sqrt(n) for n in ns_all],
                "k:",
                linewidth=2,
                label="c * sqrt(n), anchored at the smallest n",
            )
        ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("n [vertices]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    _legend_outside(ax)
    _save(fig, outpath)


def plot_p1_iterations_vs_n(stageb_rows, outpath):
    _plot_stageb_metric(
        stageb_rows,
        outpath,
        lambda r: r["iterations"],
        "iterations [count], median over graphs",
        "P1: Iterations vs n (Gabow, empty starting matching)",
        reference=True,
    )


def plot_p2_ops_per_iteration_over_m(stageb_rows, outpath):
    _plot_stageb_metric(
        stageb_rows,
        outpath,
        lambda r: (
            (r["total_ops"] / r["iterations"]) / r["m"]
            if r["iterations"] > 0 and r["m"] > 0
            else None
        ),
        "operations per iteration / m [dimensionless]",
        "P2: Work per iteration, normalized by m (expect flat)",
    )


def plot_p3_total_ops_over_bound(stageb_rows, outpath):
    _plot_stageb_metric(
        stageb_rows,
        outpath,
        lambda r: r["total_ops"] / (math.sqrt(r["n"]) * r["m"]) if r["m"] > 0 else None,
        "total operations / (sqrt(n) * m) [dimensionless]",
        "P3: Total work / (sqrt(n) * m) (expect flat or decreasing)",
    )


def _stagec_rows_for(stagec_rows, fam, variant, n=None):
    if variant == "edmonds":
        ok = lambda r: r["algorithm"] == "edmonds"
    else:
        gi = variant == "greedy_on"
        ok = lambda r: r["algorithm"] == "gabow" and r["greedy_init"] is gi
    return [
        r
        for r in stagec_rows
        if r["family"] == fam and ok(r) and (n is None or r["n"] == n)
    ]


def _plot_p4_panel(ax, stagec_rows, families, title):
    all_ns = set()
    for fam in families:
        _color, marker = _family_style(fam)
        for variant in ("greedy_on", "greedy_off", "edmonds"):
            sub = _stagec_rows_for(stagec_rows, fam, variant)
            if not sub:
                continue
            ns = sorted({r["n"] for r in sub})
            all_ns.update(ns)
            ys = [_point_time([r for r in sub if r["n"] == n]) for n in ns]
            color, ls, _m, _label = VARIANT_STYLE[variant]
            ax.plot(ns, ys, marker=marker, linestyle=ls, color=color, markersize=5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    _tidy_log_x(ax, all_ns)
    ax.set_xlabel("n [vertices]")
    ax.set_ylabel("wall-clock time [s], median over graphs")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    handles = [
        Line2D([], [], color=c, linestyle=ls, label=label)
        for key, (c, ls, _m, label) in VARIANT_STYLE.items()
        if key != "edgescan"
    ] + [
        Line2D(
            [],
            [],
            color="gray",
            marker=_family_style(f)[1],
            linestyle="none",
            label=_family_label(f),
        )
        for f in families
    ]
    _legend_outside(ax, handles=handles)


def plot_p4_wallclock_loglog(stagec_rows, outpath):
    """Wall-clock time vs n: color and line style = variant, marker = family."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6))
    fams = {r["family"] for r in stagec_rows}
    _plot_p4_panel(
        ax1,
        stagec_rows,
        _sorted_families(f for f in fams if f.startswith("F1")),
        "P4a: Wall-clock time, F1 random sparse graphs",
    )
    _plot_p4_panel(
        ax2,
        stagec_rows,
        _sorted_families(f for f in fams if f.startswith("F2")),
        "P4b: Wall-clock time, F2 random dense graphs",
    )
    fig.tight_layout()
    _save(fig, outpath)


def plot_p5_normalized_time(stagec_rows, outpath):
    """Gabow (greedy ON, the default) time / (sqrt(n) * m) and Edmonds
    time / n^3, per family."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    for fam in _sorted_families({r["family"] for r in stagec_rows}):
        color, marker = _family_style(fam)
        sub_g = _stagec_rows_for(stagec_rows, fam, "greedy_on")
        sub_e = _stagec_rows_for(stagec_rows, fam, "edmonds")
        if sub_g:
            ns = sorted({r["n"] for r in sub_g})
            ys = []
            for n in ns:
                group = [r for r in sub_g if r["n"] == n]
                m = statistics.median([r["m"] for r in group])
                ys.append(
                    _point_time(group) / (math.sqrt(n) * m) if m else float("nan")
                )
            ax1.plot(ns, ys, marker=marker, color=color, markersize=5)
        if sub_e:
            ns = sorted({r["n"] for r in sub_e})
            ys = [_point_time([r for r in sub_e if r["n"] == n]) / n**3 for n in ns]
            ax2.plot(
                ns,
                ys,
                marker=marker,
                color=color,
                markersize=5,
                label=_family_label(fam),
            )
    for ax, title, ylabel in [
        (
            ax1,
            "Gabow (greedy ON): time / (sqrt(n) * m)",
            "time / (sqrt(n) * m) [s]",
        ),
        (ax2, "Edmonds: time / n^3", "time / n^3 [s]"),
    ]:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("n [vertices]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
    _legend_outside(ax2)
    fig.suptitle("P5: Wall-clock time divided by the theoretical bound (expect flat)")
    fig.tight_layout()
    _save(fig, outpath)


def _plot_speedup_series(ax, series):
    """series: {family: [(n, (median, min, max, k)), ...]}; draws lines and,
    for bimodal points, per-graph min-max bars."""
    any_bars = False
    families = _sorted_families(series)
    for k, fam in enumerate(families):
        pts = sorted(series[fam])
        color, marker = _family_style(fam)
        # Small per-family x offset (3% per step on the log axis) so that
        # min-max bars of families measured at the same n do not overlap.
        shift = 1.03 ** (k - (len(families) - 1) / 2)
        xs = [n * shift for n, _sp in pts]
        ys = [sp[0] for _n, sp in pts]
        ax.plot(
            xs, ys, marker=marker, color=color, label=_family_label(fam), markersize=5
        )
        for n, sp in pts:
            if _is_bimodal(sp):
                any_bars = True
                ax.errorbar(
                    [n * shift],
                    [sp[0]],
                    yerr=[[sp[0] - sp[1]], [sp[2] - sp[0]]],
                    fmt="none",
                    ecolor=color,
                    elinewidth=1.2,
                    capsize=4,
                    alpha=0.8,
                )
    ax.axhline(1.0, color="black", linewidth=1, linestyle=":", label="speedup = 1")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n [vertices]")
    ax.set_ylabel("speedup = Edmonds time / Gabow time [x]")
    ax.grid(True, which="both", alpha=0.3)
    handles, _labels = ax.get_legend_handles_labels()
    if any_bars:
        handles.append(_bars_proxy())
    _legend_outside(ax, handles=handles)


def _stagec_speedup_series(stagec_rows, variant):
    series = {}
    for fam in {r["family"] for r in stagec_rows}:
        e_all = _stagec_rows_for(stagec_rows, fam, "edmonds")
        g_all = _stagec_rows_for(stagec_rows, fam, variant)
        for n in sorted({r["n"] for r in e_all} & {r["n"] for r in g_all}):
            sp = _point_speedup(
                [r for r in e_all if r["n"] == n], [r for r in g_all if r["n"] == n]
            )
            if sp:
                series.setdefault(fam, []).append((n, sp))
    return series


def _hard_rows_for(hard_rows, fam, variant, n=None):
    if variant == "edmonds":
        ok = lambda r: r["algorithm"] == "edmonds"
    else:
        ok = lambda r: r["algorithm"] == "gabow" and r["variant"] == variant
    return [
        r
        for r in hard_rows
        if r["family"] == fam and ok(r) and (n is None or r["n"] == n)
    ]


def plot_p6_speedup(stagec_rows, outpath, greedy_init=True):
    """Per-graph speedup of Gabow over Edmonds, F1/F2 (P6: greedy ON, the
    default; P6b: greedy OFF)."""
    variant = "greedy_on" if greedy_init else "greedy_off"
    fig, ax = plt.subplots(figsize=(8, 5.5))
    _plot_speedup_series(ax, _stagec_speedup_series(stagec_rows, variant))
    name = "P6" if greedy_init else "P6b"
    mode = "greedy ON" if greedy_init else "greedy OFF"
    ax.set_title(f"{name}: Speedup of Gabow ({mode}) over Edmonds, median over graphs")
    _save(fig, outpath)


def plot_p8_hard_families(hard_rows, outpath):
    """One panel per hard family: wall-clock time vs n for each Gabow
    variant and Edmonds."""
    families = _sorted_families({r["family"] for r in hard_rows})
    fig, axes = plt.subplots(1, len(families), figsize=(5.2 * len(families), 4.8))
    if len(families) == 1:
        axes = [axes]
    for ax, family in zip(axes, families):
        all_ns = set()
        for variant in ("greedy_on", "greedy_off", "edgescan", "edmonds"):
            sub = _hard_rows_for(hard_rows, family, variant)
            if not sub:
                continue
            ns = sorted({r["n"] for r in sub})
            all_ns.update(ns)
            ys = [_point_time([r for r in sub if r["n"] == n]) for n in ns]
            color, ls, marker, _label = VARIANT_STYLE[variant]
            ax.plot(ns, ys, marker=marker, linestyle=ls, color=color, markersize=6)
            for x, y in zip(ns, ys):
                if any(r.get("idle_rerun") for r in sub if r["n"] == x):
                    ax.annotate(
                        "idle re-run",
                        (x, y),
                        textcoords="offset points",
                        xytext=(8, -4),
                        fontsize=7,
                        color=color,
                    )
        ax.set_xscale("log")
        ax.set_yscale("log")
        _tidy_log_x(ax, all_ns)
        ax.set_xlabel("n [vertices]")
        ax.set_ylabel("wall-clock time [s]")
        ax.set_title(_family_label(family))
        ax.grid(True, which="both", alpha=0.3)
    handles = [
        Line2D([], [], color=c, linestyle=ls, marker=m, label=label)
        for c, ls, m, label in VARIANT_STYLE.values()
    ]
    fig.suptitle("P8: Hard graph families, wall-clock time of Gabow and Edmonds")
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        bbox_to_anchor=(0.5, -0.06),
        fontsize=9,
        frameon=False,
    )
    fig.tight_layout()
    _save(fig, outpath)


def plot_p9_speedup_all_families(stagec_rows, hard_rows, outpath):
    """Per-graph speedup of Gabow (greedy ON) over Edmonds for every family:
    F1/F2 from stageC_raw.csv, F3-F6 from stageC_hardfamilies.csv."""
    series = _stagec_speedup_series(stagec_rows, "greedy_on")
    for fam in {r["family"] for r in hard_rows}:
        e_all = _hard_rows_for(hard_rows, fam, "edmonds")
        g_all = _hard_rows_for(hard_rows, fam, "greedy_on")
        for n in sorted({r["n"] for r in e_all} & {r["n"] for r in g_all}):
            sp = _point_speedup(
                [r for r in e_all if r["n"] == n], [r for r in g_all if r["n"] == n]
            )
            if sp:
                series.setdefault(fam, []).append((n, sp))
    fig, ax = plt.subplots(figsize=(9, 6))
    _plot_speedup_series(ax, series)
    ax.set_title("P9: Speedup of Gabow (greedy ON) over Edmonds, all families")
    _save(fig, outpath)


def plot_p10_f5_edgescan(f5_edgescan_rows, outpath):
    """F5 chains from the C++ edge-scan start: (a) iterations vs n with the
    paper's Table 1 and the simulated C++ heuristic trigger; (b) total work
    / (sqrt(n) * m)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    rows = [r for r in f5_edgescan_rows if r["family"] == "F5_chains_edgescan"]
    ns = sorted({r["n"] for r in rows})
    iters = [next(r["iterations"] for r in rows if r["n"] == n) for n in ns]
    ops_over_bound = [
        next(r["total_ops"] for r in rows if r["n"] == n)
        / (math.sqrt(n) * next(r["m"] for r in rows if r["n"] == n))
        for n in ns
    ]
    color, ls, marker, _label = VARIANT_STYLE["edgescan"]
    slope, _i, r2, se = _fit_loglog(ns, iters)
    ax1.plot(
        ns,
        iters,
        marker=marker,
        linestyle=ls,
        color=color,
        markersize=7,
        label=f"ours, all iterations (slope {slope:.2f} +/- {se:.2f}, R2 = {r2:.2f})",
    )
    sim_n, sim_it = _simulate_heur_trigger_series(
        sizes=tuple(sorted({r["n_param"] for r in rows}))
    )
    if sim_n:
        ax1.plot(
            sim_n,
            sim_it,
            "D",
            color="tab:purple",
            markersize=9,
            label="ours, simulated C++ heuristic trigger",
        )
    ax1.plot(
        [10000, 20000, 40000],
        [24, 33, 47],
        "k*",
        markersize=14,
        label="paper, Table 1",
        zorder=5,
    )
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    _tidy_log_x(ax1, ns)
    ax1.set_yticks([20, 30, 50, 100, 200])
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}"))
    ax1.yaxis.set_minor_formatter(NullFormatter())
    ax1.set_xlabel("n [vertices]")
    ax1.set_ylabel("iterations [count]")
    ax1.set_title("P10a: Iterations vs n (expect slope 0.5)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, which="both", alpha=0.3)

    ax2.plot(ns, ops_over_bound, marker=marker, linestyle=ls, color=color, markersize=7)
    for r in rows:
        # The dip at an odd-order clique is a property of the graph (one
        # clique vertex stays free under the edge-scan start); see SUMMARY.
        clique = max(2, int(math.sqrt(4 * int(r["n_param"]))))
        if clique % 2:
            ax2.annotate(
                f"odd clique ({clique} vertices)",
                (r["n"], r["total_ops"] / (math.sqrt(r["n"]) * r["m"])),
                textcoords="offset points",
                xytext=(10, 0),
                fontsize=8,
                color=color,
            )
    ax2.set_xscale("log")
    _tidy_log_x(ax2, ns)
    ax2.set_xlabel("n [vertices]")
    ax2.set_ylabel("total operations / (sqrt(n) * m) [dimensionless]")
    ax2.set_title("P10b: Total work / (sqrt(n) * m) (expect flat)")
    ax2.grid(True, which="both", alpha=0.3)
    fig.suptitle("P10: F5 chains from the C++ edge-scan starting matching")
    fig.tight_layout()
    _save(fig, outpath)


# ---------------------------------------------------------------------------
# SUMMARY.md
# ---------------------------------------------------------------------------


def _analyze_stagec_bimodality(stagec_rows):
    """For every (family, n) with Gabow greedy_init=True timings, report
    mean, median, std, and -- the actual verification/refutation of the
    "greedy already found the maximum, so the main loop barely runs"
    hypothesis -- split the SAME runtimes by the (per-graph, not per-run)
    greedy_already_optimal diagnostic flag and compare the two groups'
    mean/median directly. If greedy_already_optimal graphs are
    systematically much faster than the rest, that confirms the
    hypothesis with data, not just a plausible story.
    """
    out = []
    keys = sorted(
        {
            (r["family"], r["n"])
            for r in stagec_rows
            if r["algorithm"] == "gabow" and r["greedy_init"] is True
        }
    )
    for family, n in keys:
        sub = [
            r
            for r in stagec_rows
            if r["family"] == family
            and r["n"] == n
            and r["algorithm"] == "gabow"
            and r["greedy_init"] is True
        ]
        vals = [r["runtime_sec"] for r in sub]
        if len(vals) < 2:
            continue
        mean = statistics.mean(vals)
        med, iqr = _median_iqr(vals)
        std = statistics.stdev(vals)
        bimodal_flag = (
            mean > 0 and (std / mean > 0.5) and (mean / med > 2 if med > 0 else True)
        )

        already_opt_graphs = {r["seed"] for r in sub if r["greedy_already_optimal"]}
        not_opt_graphs = {r["seed"] for r in sub if not r["greedy_already_optimal"]}
        vals_opt = [r["runtime_sec"] for r in sub if r["seed"] in already_opt_graphs]
        vals_not = [r["runtime_sec"] for r in sub if r["seed"] in not_opt_graphs]

        line = (
            f"- {family} n={n}: mean={mean:.4g}s median={med:.4g}s IQR={iqr:.4g}s std={std:.4g}s "
            f"({'BIMODAL' if bimodal_flag else 'not clearly bimodal'} by mean/median/std heuristic)"
        )
        if vals_opt and vals_not:
            mo, ms = statistics.mean(vals_opt), statistics.mean(vals_not)
            line += (
                f" -- greedy_already_optimal seeds: mean={mo:.4g}s (n={len(vals_opt)}); "
                f"NOT already optimal seeds: mean={ms:.4g}s (n={len(vals_not)}); "
                f"ratio={ms / mo if mo > 0 else float('inf'):.1f}x "
                f"({'CONFIRMS' if ms > 2 * mo else 'does not clearly confirm'} the greedy-init explanation)"
            )
        elif vals_opt and not vals_not:
            line += " -- ALL seeds had greedy already optimal (no contrast available at this n)"
        elif vals_not and not vals_opt:
            line += " -- NO seeds had greedy already optimal at this n (no contrast available)"
        out.append(line)
    return out


def _analyze_f5_counters_table(stageb_rows):
    """Full F5_chains_unshuffled (greedy_init=False) table across every
    n_param measured in this run (both the originally-quick sizes and the
    --full sizes, which previously lived in separate, mutually-overwriting
    CSVs -- now merged into one stageB_raw.csv). For each row, also reports
    the generator's clique size (int(sqrt(4*n_param)), from
    verify_complexity.chains_graph) and its PARITY, which this
    investigation found to be an exact predictor of whether
    blossom_contraction_events is zero: a complete graph on an EVEN number
    of vertices has a perfect internal matching, so the attachment vertex z
    gets matched entirely inside the clique and no blossom ever needs to
    form; on an ODD clique, one clique vertex is always left over, which
    forces a blossom. Verified against 29 n_param values spanning 100 to
    15000 with zero exceptions (not just the handful in this table) before
    writing this explanation.
    """
    sub = [
        r
        for r in stageb_rows
        if r["family"] == "F5_chains_unshuffled" and r["greedy_init"] is False
    ]
    ns = sorted({r["n"] for r in sub})
    out = []
    if not ns:
        return out
    out.append(
        "| n | n_param | m | iterations | blossom_contraction_events | "
        "ops/iteration/m | clique size | clique parity |"
    )
    out.append("|---|---|---|---|---|---|---|---|")
    for n in ns:
        group = [r for r in sub if r["n"] == n]
        n_param = group[0]["n_param"]
        m = group[0]["m"]
        its = statistics.median([r["iterations"] for r in group])
        bce = statistics.median([r["blossom_contraction_events"] for r in group])
        ops_m = statistics.median(
            [
                (r["total_ops"] / r["iterations"]) / r["m"]
                for r in group
                if r["iterations"] and r["m"]
            ]
        )
        clique = int(math.sqrt(4 * n_param))
        parity = "odd" if clique % 2 else "even"
        out.append(
            f"| {n} | {n_param} | {m} | {its:.0f} | {bce:.0f} | {ops_m:.3f} | {clique} | {parity} |"
        )
    out.append(
        "\nNon-monotonic, explained: `blossom_contraction_events` is NOT a "
        "smoothly growing quantity that jumps once and plateaus -- it is "
        "exactly zero whenever the clique-size column above is even, and "
        "clearly positive whenever it is odd, with no exceptions in this "
        "table or in a separate sweep of 29 n_param values from 100 to "
        "15000 checked specifically for this investigation. The three "
        "--full sizes (n_param=10000/20000/40000, clique sizes 200/282/400) "
        "happen to all be even, so they all show blossom_contraction_events=0 "
        "and the flat ops/iteration/m ~ 2.6 reported in P2's main text; the "
        "originally-quick sizes (n_param=1000/2000, clique sizes 63/89) "
        "happen to both be odd, so they show nonzero contractions and "
        "ops/iteration/m ~ 8.7. This is a parity coincidence of which "
        "three particular sizes --full happened to sample, not a trend "
        'with n -- the earlier "jump" framing (comparing just the two '
        "smallest quick-run sizes to each other) was a real but incomplete "
        "reading of this effect; it undersold how large the swing actually "
        "is (odd-clique ops/iteration/m is consistently ~3x the even-clique "
        "value, not a one-time jump) and did not mention that it reverses.\n"
    )
    return out


PAPER_SHORT_AND_LONG = [
    # (n, m, iterations) read directly off Table 1's "SHORT AND LONG" columns,
    # from the PDF (not the HTML summarizer -- see SUMMARY.md citation check).
    # n/10^3 and m/10^3 in the paper; converted to absolute counts here.
    (10000, 22000, 24),
    (20000, 56000, 33),
    (40000, 114000, 47),
]


# Paper Table 1 n (vertices) -> our generator's n_param with the same vertex
# count (our graphs have ~2*n_param vertices).
PAPER_N_TO_NPARAM = {10000: 5000, 20000: 10000, 40000: 20000}


def _f5_size_comparison(f5_edgescan_rows, sim_trigger):
    """One row per paper Table 1 row: paper (n, m, #it) next to our graph at
    the matching n_param (PAPER_N_TO_NPARAM), with our full-run iteration
    count from the edge-scan start (the C++ reference's own init()) and the
    iteration at which the C++ driver's `heur` fallback would trigger,
    simulated from our per-iteration augmentation counts.
    """
    by_nparam = {
        r["n_param"]: r for r in f5_edgescan_rows if r["family"] == "F5_chains_edgescan"
    }
    out = [
        (
            "| paper n | paper m | paper #it | n_param | our n (vertices) | our m | "
            "our iterations (full run) | simulated heur trigger |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]
    for paper_n, paper_m, paper_it in PAPER_SHORT_AND_LONG:
        n_param = PAPER_N_TO_NPARAM[paper_n]
        r = by_nparam.get(n_param)
        trig = sim_trigger.get(n_param, "--")
        if r is None:
            out.append(
                f"| {paper_n} | {paper_m} | {paper_it} | {n_param} | -- | -- | -- | {trig} |"
            )
        else:
            out.append(
                f"| {paper_n} | {paper_m} | {paper_it} | {n_param} | {r['n']} | "
                f"{r['m']} | {r['iterations']} | {trig} |"
            )
    return out


def write_summary(
    stageb_rows,
    stagec_rows,
    env_info,
    elapsed,
    path,
    stable_rows=None,
    f5_edgescan_rows=None,
    hard_rows=None,
):
    lines = []
    lines.append("# Stage B/C experimental summary\n")
    lines.append(
        "Experimental support (not proof) for max_cardinality_matching_gabow "
        "running in O(sqrt(n) * m * alpha(n)), and a wall-clock comparison "
        "against NetworkX's max_weight_matching(maxcardinality=True) "
        "(Edmonds, O(n**3)). See docs/complexity_audit.md for the static, "
        "code-level half of this claim (per-iteration cost); this document "
        "covers the empirical half (iteration count, and wall-clock).\n"
    )
    lines.append("## Environment\n")
    for k, v in env_info.items():
        lines.append(f"- **{k}**: {v}")
    lines.append(f"- **total run time**: {elapsed:.1f}s\n")

    def fam_iter_summary(prefix):
        fams = sorted(
            {
                r["family"]
                for r in stageb_rows
                if r["family"].startswith(prefix) and r["greedy_init"] is False
            }
        )
        out = []
        for fam in fams:
            sub = [
                r
                for r in stageb_rows
                if r["family"] == fam and r["greedy_init"] is False
            ]
            ns = sorted({r["n"] for r in sub})
            if len(ns) < 2:
                continue
            its = [
                statistics.median([r["iterations"] for r in sub if r["n"] == n])
                for n in ns
            ]
            slope, _i, r2, _se = _fit_loglog(ns, its)
            out.append(f"{fam}: iterations ~ n^{slope:.2f} (R2={r2:.2f}) over n={ns}")
        return out

    lines.append("## Plots\n")
    lines.append(
        "P1 iterations vs n; P2 work per iteration / m; P3 total work / "
        "(sqrt(n) * m); P4 wall-clock time; P5 wall-clock time / bound; P6 / "
        "P6b speedup over Edmonds (greedy ON / OFF); P8 hard families vs "
        "Edmonds; P9 speedup, all families; P10 F5 with the C++ edge-scan "
        "start. P7 was removed: P10 supersedes it.\n"
    )
    lines.append("## P1 -- Iterations vs n\n")
    lines.append(
        "Plots median iterations (greedy_init=OFF) against n for every Stage "
        "B family, on log-log axes, with a c*sqrt(n) reference curve anchored "
        "at the SMALLEST n measured (not the largest -- anchoring at the "
        "largest point would let a flat family look deceptively close to "
        "the reference everywhere; anchoring at the smallest point means a "
        "family that actually tracks sqrt(n) growth will visibly follow the "
        "line, while a flat family will visibly fall below it as n grows). "
    )
    for s in fam_iter_summary("F"):
        lines.append(f"- {s}")
    lines.append(
        "\n**Verdict**: supports the O(sqrt(n)) claim only where the fitted "
        "exponent is close to 0.5; an exponent near 0 (flat) means that "
        "family resolves in few iterations regardless of n and gives no "
        "evidence either way about tightness.\n"
    )

    lines.append("## P2 -- Ops per iteration / m\n")
    lines.append(
        "(total_ops / iterations) / m per family, expected flat if each "
        "iteration really costs O(m * alpha(n)) (the O(1)-ish alpha(n) "
        "factor should not visibly grow with n at these scales).\n"
        "**Verdict**: supports the per-iteration bound if flat; a rising "
        "trend would suggest the per-iteration cost is growing faster than "
        "m, which the static audit (docs/complexity_audit.md) did not find "
        "any mechanism for.\n"
    )
    lines.append("### F5 counters across all measured sizes\n")
    f5_jump = _analyze_f5_counters_table(stageb_rows)
    if f5_jump:
        lines.extend(f5_jump)
    else:
        lines.append("- Not enough F5 sizes in this run to investigate the jump.")
    lines.append("")

    lines.append("## P3 -- Total ops / bound(n,m)\n")
    lines.append(
        "total_ops / (sqrt(n) * m) per family. Flat or decreasing supports "
        "the full bound empirically; a clearly increasing trend would be "
        "the strongest possible experimental evidence *against* it (though "
        "still not a disproof, since n here is finite).\n"
    )

    lines.append("## Stage C repetition policy\n")
    lines.append(
        f"Each (family, n, seed, algorithm) point uses >= {MIN_REPEATS} "
        f"repeated timed calls on the same graph, or until {TIME_BUDGET_SEC}s "
        f"total elapses, whichever is reached last -- EXCEPT once a single "
        f"call is measured to take longer than {SLOW_CALL_REPEAT_THRESHOLD_SEC}s, "
        "in which case only that ONE call is used (no further within-seed "
        "repeats): for slow calls (this affects F1's Edmonds comparison in "
        f"particular, at n near the {TIMEOUT_SEC}s-calibrated cap), the "
        f"intended statistical spread comes from the >= {len(SEEDS)} "
        "different seeds run at that point instead of repeating the same "
        "graph many times at tens of seconds each. Rows derived from a "
        "single-repetition point are directly identifiable in "
        "stageC_raw.csv (exactly one row for that family/n/seed/algorithm "
        "combination instead of several).\n"
    )

    lines.append("## P4 -- Wall-clock, Gabow (greedy ON/OFF) vs Edmonds (log-log)\n")
    lines.append(
        "Two panels, F1 (sparse) and F2 (dense). Color and line style give "
        "the variant (Gabow greedy ON / OFF, Edmonds), the marker gives the "
        "family. Each point is the median over graphs of each graph's median "
        "time. Fitted log-log slopes:\n"
    )
    for fam in sorted({r["family"] for r in stagec_rows}):
        for algo, greedy_init in (("gabow", True), ("gabow", False), ("edmonds", None)):
            sub = [
                r
                for r in stagec_rows
                if r["family"] == fam
                and r["algorithm"] == algo
                and r["greedy_init"] == greedy_init
            ]
            if not sub:
                continue
            ns = sorted({r["n"] for r in sub})
            if len(ns) < 2:
                continue
            meds = [_point_time([r for r in sub if r["n"] == n]) for n in ns]
            slope, _i, r2, se = _fit_loglog(ns, meds)
            tag = f"{algo}/greedy={greedy_init}" if algo == "gabow" else algo
            lines.append(
                f"- {fam}/{tag}: time ~ n^{slope:.2f} +/- {se:.2f} (R2={r2:.2f})"
            )
    lines.append(
        "\n**Verdict**: compare fitted slopes above to the bound-implied "
        "reference (sparse: Gabow~1.5, dense: Gabow~2.5, Edmonds~3 "
        "regardless of density). Measured slopes below the reference at "
        "modest n are expected (not yet asymptotic), not a contradiction.\n"
    )

    lines.append("### F1 Edmonds coverage (c=3 vs c=5 vs c=10)\n")
    lines.append(
        "An earlier version of this script keyed `edmonds_disabled` by "
        '`fam_key` ("F1", from `family.split("_")[0]`) instead of the '
        "full family string. F1_sparse_c3's single measured Edmonds call at "
        "n=20000 (seed=0) took 258.5s, over the 120s disable-threshold, "
        'which set `edmonds_disabled["F1"]=True` -- and because that one '
        "flag was shared across c=3/c=5/c=10, it silently skipped Edmonds "
        "for F1_sparse_c5 and F1_sparse_c10 at EVERY size, including n=1000 "
        "and n=5000 where Edmonds is fast and was never actually slow for "
        "those c-values. This was a benchmark-harness bug, not a reflection "
        "of Edmonds' real performance at those sizes -- fixed by keying the "
        "disable flag on the full family name instead (each c-value now "
        "disables independently), and a supplementary run collected the "
        "missing data without re-running the full suite:\n"
        "- F1_sparse_c5: n=1000 and n=5000, all 5 seeds; n=20000, seed=0 "
        "only (350.3s, over the 120s threshold, correctly disabled for "
        "larger n in this family afterward -- same policy as c=3).\n"
        "- F1_sparse_c10: n=1000 and n=5000, all 5 seeds; n=20000, seed=0 "
        "only (166.7s, same reasoning).\n"
        "- F1_sparse_c3's own existing data is unaffected by the fix: its "
        "n=20000 point was legitimately down to 1 seed already (its own "
        "seed=0 call was the one that exceeded 120s), not a side-effect of "
        "the bug.\n"
    )
    lines.append(
        "\n**69s calibration vs ~260s measured, explained**: "
        "`_calibrate_f1_edmonds_cap` timed ONE untimed Edmonds call on "
        "f1_random_sparse(20000, 3, seed=0) before Stage C started and got "
        "69.3s; Stage C's own timed call on the SAME construction "
        "(identical n, c, seed -- nx.gnm_random_graph is deterministic "
        "given a seed, so this is provably the same graph and the same "
        "deterministic algorithm) got 258.5s, a 3.7x difference with no "
        "difference in input. The two calls differ only in WHEN they ran: "
        "the calibration ran first, on an otherwise-idle machine; the Stage "
        "C call ran roughly an hour into the same --full process, after "
        "accumulating Stage B/C state (rows, matplotlib figures, GC "
        "pressure) and whatever else was happening on this machine at the "
        "time. Because a call this slow only gets ONE repetition (the "
        "policy documented above), there is no within-point averaging to "
        "smooth this out -- single-sample timings of multi-minute calls on "
        "a shared, non-isolated machine should be read as having a wide "
        "and not-well-characterized error bar, not as a precise number. "
        "F1_sparse_c5 (350.3s) and F1_sparse_c10 (166.7s) at the same n and "
        "seed=0 show the same kind of spread with no consistent trend "
        "against c, consistent with this being measurement noise rather "
        "than a real c-dependent effect.\n"
        "\n**All three F1 n=20000 Edmonds points (c=3, c=5, c=10) are "
        "SINGLE-SAMPLE measurements (258.5s, 350.3s, 166.7s respectively) "
        "and should be read as noisy** -- the P4a/P6/P6b points at "
        "n=20000 rest on that one graph (seed 0) and inherit that "
        "uncertainty; they could plausibly be off by a factor of several. "
        "See the stable, repeated measurement at n=10000 directly below for "
        "a number with real error bars.\n"
    )

    lines.append("### Stable F1 Edmonds check (n=10000, repeated, isolated process)\n")
    if stable_rows:
        lines.append(
            "Run via `run_f1_stable_check()` in a dedicated, freshly-started "
            f"Python process (no other benchmark running), on the final code "
            f"(FIFO bucket queue), at "
            f"n={STABLE_CHECK_N} (smaller than the n=20000 points above "
            "specifically so Edmonds is cheap enough to repeat for real): "
            f"edmonds x{STABLE_CHECK_EDMONDS_REPEATS}, gabow (both greedy "
            f"modes) x{STABLE_CHECK_GABOW_REPEATS}, for c in "
            f"{list(STABLE_CHECK_CS)} and seeds {list(STABLE_CHECK_SEEDS)}. "
            "Each matching was also checked against Gabow's for every "
            "(c, seed, greedy mode), per the assertion in the collection "
            "code (an unmatched pair aborts the run rather than silently "
            "recording wrong data).\n"
        )
        lines.extend(_stable_check_table(stable_rows))
        lines.append(
            "\nCompare to the single-sample n=20000 points above (e.g. c=3: "
            "edmonds=258.5s in one call): the n=10000 numbers here are "
            "medians of repeated runs and give the order of magnitude of the "
            "speedup without relying on a single noisy multi-minute call.\n"
        )
    else:
        lines.append(
            "- Not run in this invocation (pass `stable_rows` from "
            "`run_f1_stable_check()`, or run `python run_all.py "
            "--stable-check`, to populate this section).\n"
        )

    lines.append("## Stage C: bimodal-runtime investigation (dense graphs)\n")
    lines.append(
        "Reported mean and median (not just median) per (family, n), for "
        'Gabow greedy_init=True, and a direct test of the "greedy init '
        'alone already found the maximum matching" explanation: the SAME '
        "runtimes are split by the per-graph `greedy_already_optimal` "
        "diagnostic flag (augmentations == 0, i.e. the only iteration run "
        "was the single final search confirming nothing more to do) and "
        "the two groups' means are compared directly.\n"
    )
    bimodal_lines = _analyze_stagec_bimodality(stagec_rows)
    if bimodal_lines:
        lines.extend(bimodal_lines)
    else:
        lines.append(
            "- No (family, n) with >= 2 Gabow greedy_init=True runs in this run."
        )
    lines.append(
        "\n**Verdict**: where a (family, n) is marked BIMODAL and the "
        "greedy_already_optimal-seed mean is much lower than the "
        "not-already-optimal-seed mean, the greedy-init explanation is "
        "CONFIRMED by data for that point, not just plausible. Where only "
        "one group exists at a given n (all-optimal or none-optimal), "
        "there is no contrast to test the explanation against at that "
        "specific point.\n"
    )

    lines.append("## P5 -- Normalized wall-clock\n")
    lines.append(
        "Gabow (greedy_init=True only -- the real, user-facing function; "
        "see the Stage C bimodality section above for greedy ON vs OFF) "
        "time/(sqrt(n)*m) and Edmonds time/n^3, per family. Flat or "
        "decreasing supports the respective bound on wall-clock time "
        "directly (not just operation counts).\n"
    )

    lines.append("## P6, P6b, P9 -- Speedup of Gabow over Edmonds\n")
    lines.append(
        "How each point is computed: for every graph, the median of its "
        "timed runs, separately for Edmonds and for Gabow; that graph's "
        "speedup is the ratio Edmonds / Gabow; the point is the median of "
        "these per-graph speedups, with their min-max over graphs in "
        "brackets. A plain median over all timed runs (used before) "
        "over-weights easy graphs, because fast calls are repeated until "
        f"{TIME_BUDGET_SEC}s elapse. F3-F6 build one fixed graph per n, so "
        "all their runs count as one graph. A point whose per-graph speedups "
        f"span more than {BIMODAL_SPREAD:g}x is marked bimodal; P6, P6b and "
        "P9 draw its min-max range.\n"
    )
    lines.extend(_speedup_table(stagec_rows))
    on = _stagec_speedup_series(stagec_rows, "greedy_on")
    off = _stagec_speedup_series(stagec_rows, "greedy_off")
    on_pts = [(f, n, sp) for f, pts in on.items() for n, sp in pts]
    off_pts = [(f, n, sp) for f, pts in off.items() for n, sp in pts]
    on_bimodal = [(f, n) for f, n, sp in on_pts if _is_bimodal(sp)]
    on_slow = sorted((f, n) for f, n, sp in on_pts if sp[0] < 1)
    off_f2 = [(f, n, sp) for f, n, sp in off_pts if f.startswith("F2")]
    off_f2_slow = [(f, n) for f, n, sp in off_f2 if sp[0] < 1]
    lines.append(
        f"\n- Greedy ON: {len(on_bimodal)} of {len(on_pts)} points are "
        "bimodal (on some graphs the greedy matching is already maximum and "
        "Gabow finishes almost at once, on others it runs the full search; "
        "see the Stage C bimodality section). Points with median speedup "
        f"below 1: {', '.join(f'{f} n={n}' for f, n in on_slow) or 'none'}."
    )
    lines.append(
        f"- Greedy OFF: Gabow is slower than Edmonds (speedup < 1) at "
        f"{len(off_f2_slow)} of {len(off_f2)} F2 points; every F1 point is "
        "a speedup > 1."
        if all(sp[0] > 1 for f, n, sp in off_pts if f.startswith("F1"))
        else f"- Greedy OFF: Gabow is slower than Edmonds (speedup < 1) at "
        f"{len(off_f2_slow)} of {len(off_f2)} F2 points."
    )
    lines.append(
        "\nNote on F2_density100: greedy init alone was already the maximum "
        "matching on every measured graph at every n with greedy ON "
        "(augmentations == 0 throughout), so its greedy-ON line measures the "
        "cost of the greedy pass itself, not of Gabow's search.\n"
    )

    lines.append("## Dense graphs, greedy OFF vs Edmonds (plain result, not a bug)\n")
    dense = []
    for d in (10, 25, 50, 75, 100):
        fam = f"F2_density{d}"
        e = _stagec_rows_for(stagec_rows, fam, "edmonds", 800)
        g_on = _stagec_rows_for(stagec_rows, fam, "greedy_on", 800)
        g_off = _stagec_rows_for(stagec_rows, fam, "greedy_off", 800)
        if e and g_on and g_off:
            dense.append((fam, d, e, g_on, g_off, _point_speedup(e, g_off)))
    slower = [f"{d}%" for _f, d, _e, _on, _off, sp in dense if sp[0] < 1]
    lines.append(
        "P4b shows this as slopes; here with numbers at n=800 (per-graph "
        "medians, then the median over graphs). With greedy OFF (an empty "
        "starting matching, as in the paper), Gabow is slower than Edmonds "
        f"at density {', '.join(slower) or 'none'} -- a constant factor of "
        "this Python implementation when it must find every augmenting path "
        "from scratch, not evidence against the O(sqrt(n) * m * alpha(n)) "
        "bound, which compares Gabow with its own n and m, not with "
        "Edmonds' constant.\n"
    )
    for fam, _d, e, g_on, g_off, sp_off in dense:
        sp_on = _point_speedup(e, g_on)
        lines.append(
            f"- {fam}, n=800: Edmonds {_point_time(e):.4g}s, Gabow greedy OFF "
            f"{_point_time(g_off):.4g}s, greedy ON {_point_time(g_on):.4g}s; "
            f"speedup OFF {_fmt_speedup(sp_off)}, ON {_fmt_speedup(sp_on)}"
        )
    lines.append(
        "\nSee plots/P6b_speedup_greedy_off.png (greedy OFF) against "
        "plots/P6_speedup.png (greedy ON, the default).\n"
    )

    lines.append("## P8/P9 -- Hard families (F3-F6) vs Edmonds\n")
    if hard_rows:
        lines.append(
            "Wall-clock per (family, n); each of these families builds one "
            "fixed graph per n, so times are medians over its runs. `n` is the "
            "actual vertex count. Edmonds is capped: once one call exceeds "
            f"{HARD_EDMONDS_TIMEOUT_SEC:.0f}s it is disabled for larger n in that "
            "family (`k` = number of Edmonds samples; k=1 points are single, "
            "noisy measurements). F5's `edgescan` variant starts from the C++ "
            "reference's edge-scan greedy in true insertion order (same as P10). "
            "Rows marked `idle re-run` come from a separate re-measurement "
            f"({F5_IDLE_RERUN_CSV}) in one process with no other benchmark "
            "running, which replaces the original edge-scan and Edmonds "
            "timings at that size (taken while other work was running, with a "
            "single Edmonds sample).\n"
        )
        lines.append(
            "| family | n (vertices) | n_param | variant | iterations | gabow (s) | edmonds (s) | k | speedup | note |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for fam, n in sorted({(r["family"], r["n"]) for r in hard_rows}):
            sub = [r for r in hard_rows if r["family"] == fam and r["n"] == n]
            e_rows = [r for r in sub if r["algorithm"] == "edmonds"]
            e = [r["runtime_sec"] for r in e_rows]
            e_med = _point_time(e_rows) if e_rows else None
            for variant in ("greedy_on", "greedy_off", "edgescan"):
                g = [
                    r
                    for r in sub
                    if r["algorithm"] == "gabow" and r["variant"] == variant
                ]
                if not g:
                    continue
                g_med = _point_time(g)
                sp = _point_speedup(e_rows, g) if e_rows else None
                it = statistics.median(r["iterations"] for r in g)
                g_idle = any(r.get("idle_rerun") for r in g)
                e_idle = any(
                    r.get("idle_rerun") for r in sub if r["algorithm"] == "edmonds"
                )
                if g_idle:
                    note = f"idle re-run ({len(g)} Gabow / {len(e)} Edmonds runs)"
                elif e_idle:
                    note = f"Edmonds from idle re-run ({len(e)} runs)"
                else:
                    note = ""
                lines.append(
                    f"| {fam} | {n} | {g[0]['n_param']} | {variant} | {it:g} | {g_med:.4g} | "
                    + (
                        f"{e_med:.4g} | {len(e)} | {sp[0]:.1f}x |"
                        if e
                        else "-- | 0 | -- |"
                    )
                    + f" {note} |"
                )
        lines.append("")
    else:
        lines.append(
            "- Not run in this invocation (`python run_all.py --hard-families`).\n"
        )

    lines.append("## P10 -- F5 with the C++ reference's own initial matching\n")
    if f5_edgescan_rows:
        lines.append(
            "An investigation into why F5 (greedy OFF or ON) did not reproduce the "
            "paper's Table 1 iteration counts found the cause: a simulation of "
            "the C++ heuristic trigger reproduces Table 1 (24/33/47) exactly at "
            "all three sizes (see the Table 1 comparison below). The iterations "
            "it counts start "
            "from a different initial matching: the companion-page C++ reference (GabowBeautified.h's `init()`, "
            "identical in GabowRevised.h; called "
            "unconditionally before the phase loop in the driver that produced "
            "Table 1) uses a GLOBAL EDGE-SCAN greedy -- scan edges in insertion "
            "order, match an edge if both endpoints are still free -- which is a "
            "different algorithm from this function's own vertex-scan greedy "
            "(`for v in G: match v to its first free neighbor`). On this "
            "specific chain structure, the two produce qualitatively different "
            "starting matchings: the edge-scan greedy leaves exactly the chain "
            "endpoints exposed that the paper's O(sqrt(n))-iterations argument "
            "depends on, while the vertex-scan greedy happens to perfectly "
            "match every chain in one pass, trivially pre-solving this family. "
            "This is reproduced here through `max_cardinality_matching_gabow`'s "
            "`_initial_mate` parameter (added for this purpose, not part of the "
            "public API -- see its docstring), not a scratch copy.\n"
        )
        rows = [r for r in f5_edgescan_rows if r["family"] == "F5_chains_edgescan"]
        ns = sorted({r["n"] for r in rows})
        lines.append(
            "Units: `n` below (and P10's x-axis) is the ACTUAL vertex count "
            "G.number_of_nodes() for our points; the paper's Table 1 `n` is "
            "likewise a vertex count (its m values match our graphs at the "
            "same vertex counts; see the Table 1 comparison below), so both are "
            "plotted on the same axis. `n_param` is only our generator's "
            "construction input.\n"
        )
        lines.append(
            "| n_param | n (vertices) | m | clique size | iterations | total_ops | ops/(sqrt(n)*m) |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for n in ns:
            r = next(r for r in rows if r["n"] == n)
            ops_bound = r["total_ops"] / (math.sqrt(r["n"]) * r["m"])
            clique = max(2, int(math.sqrt(4 * int(r["n_param"]))))
            lines.append(
                f"| {r['n_param']} | {r['n']} | {r['m']} | {clique} | {r['iterations']} | "
                f"{r['total_ops']} | {ops_bound:.4f} |"
            )
        lines.append("| -- | 10000 | 22000 | -- | 24 (paper) | -- | -- |")
        lines.append("| -- | 20000 | 56000 | -- | 33 (paper) | -- | -- |")
        lines.append("| -- | 40000 | 114000 | -- | 47 (paper) | -- | -- |")
        lines.append(
            "\n**The n_param=5000 ops outlier (~0.35 vs ~3.2) is a property of "
            "the graph, not a script or algorithm bug**: it is the only size "
            "whose clique has ODD order (floor(sqrt(4*n_param)) = 141), so the "
            "edge-scan greedy leaves one clique vertex free; each iteration's "
            "search then does ~5x less edge-scan work (median per-iteration "
            "edge_scans 5283 vs ~24000 at the even-clique neighbors n_param=4900/"
            "5100), while the iteration count is unaffected (69, on the "
            "sqrt(n) trend). Reproduced at another odd clique (n_param=5150, "
            "clique 143: 0.352). Same parity effect as the F5 note under P2.\n"
        )
        lines.append(
            "\nFootnote (clique size): our generator uses clique size "
            "floor(sqrt(m)) with m = 4*n_param, a construction parameter (not "
            "the graph's edge count), exactly as the companion "
            "mc_timing_long_chains.cpp / mc_timing_short_chains.cpp "
            "(`int n = (int) sqrt(m);`), so odd cliques occur there too "
            "(n_param=5000 -> 141). KurtTest.cpp and the ADM24 text use "
            "`int n = 2* (int) sqrt(m/2.0);` instead (always even, about "
            "sqrt(2m) vertices: 140/200/282/400/564 for n_param=2500..40000). "
            "Table 1's m column (56k/114k at n=20k/40k) matches the "
            "sqrt(m) version (ours: 56,970/114,351), not the sqrt(2m) one "
            "(76,691/154,530), so ours is kept.\n"
        )
        slope, _i, r2, se = _fit_loglog(
            ns, [next(r["iterations"] for r in rows if r["n"] == n) for n in ns]
        )
        lines.append(
            f"\nFitted log-log slope of iterations vs n: {slope:.2f} +/- {se:.2f} "
            f"(R2={r2:.2f}); the O(sqrt(n)) bound predicts ~0.5. See "
            "plots/P10_f5_edgescan.png for both this fit (left panel, with the "
            "paper's points overlaid) and total_ops/(sqrt(n)*m) vs n (right "
            "panel, expected flat if the per-iteration O(m*alpha(n)) bound "
            "holds).\n"
        )
        lines.append(
            "\n**Plainly stated**: with the C++ reference's own initial "
            "matching, the iteration count on this family grows like sqrt(n) "
            "and the total work grows like sqrt(n)*m, i.e. the O(sqrt(n)) "
            "iteration bound is attained up to a constant factor on this "
            "family -- it is not merely an upper bound that this "
            "reconstruction failed to stress. Our default vertex-scan greedy "
            "warm start happens to solve this entire family in a single "
            "iteration, which is a property of that specific graph family and "
            "greedy algorithm pairing, not evidence against the bound itself.\n"
        )
        sim_trigger = {}
        if f5_edgescan_rows:
            ed_rows = [
                r for r in f5_edgescan_rows if r["family"] == "F5_chains_edgescan"
            ]
            n_params = tuple(sorted({r["n_param"] for r in ed_rows}))
            sim_n, sim_it = _simulate_heur_trigger_series(sizes=n_params)
            n_to_nparam = {r["n"]: r["n_param"] for r in ed_rows}
            sim_trigger = {n_to_nparam[n]: it for n, it in zip(sim_n, sim_it)}
        paper_its = [it for _n, _m, it in PAPER_SHORT_AND_LONG]
        sim_its = [
            sim_trigger.get(PAPER_N_TO_NPARAM[n]) for n, _m, _it in PAPER_SHORT_AND_LONG
        ]
        if sim_its == paper_its:
            sim_verdict = (
                "the simulated trigger iteration reproduces Table 1's "
                f"{'/'.join(map(str, paper_its))} EXACTLY"
            )
        else:
            sim_verdict = (
                f"the simulated trigger iterations are {sim_its} against Table 1's "
                f"{paper_its} (NOT an exact match)"
            )
        lines.append("### Comparison with the paper's Table 1\n")
        lines.append(
            "Each paper row is matched to the n_param with the same vertex "
            "count (paper n=10000/20000/40000 -> n_param=5000/10000/20000). "
            "Table 1's #it counts iterations until the C++ driver's `heur` "
            "fallback triggers, starting from the C++ edge-scan greedy "
            "(GabowBeautified.h's `init()`, identical in GabowRevised.h); the "
            "last column simulates that trigger on our per-iteration "
            f"augmentation counts: {sim_verdict}.\n"
        )
        lines.extend(_f5_size_comparison(f5_edgescan_rows or [], sim_trigger))
        lines.append("")
    else:
        lines.append(
            "- Not run in this invocation (pass `f5_edgescan_rows` from "
            "`run_f5_edgescan()`, or run `python run_all.py --f5-edgescan`, "
            "to populate this section).\n"
        )

    lines.append(
        "\n---\n\n**Overall**: this document reports experimental support, "
        "not proof. See docs/complexity_audit.md for what is verified "
        "directly from the code (per-iteration cost) versus what remains "
        "cited theory or open (the O(sqrt(n)) iteration count, and its "
        "tightness).\n"
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _ram_bytes():
    """Total physical RAM, without adding a new dependency: ctypes on
    Windows, /proc/meminfo on Linux, sysctl on macOS; "unknown" if none of
    these apply (e.g. an unsupported platform).
    """
    try:
        if sys.platform == "win32":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullTotalPhys
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) * 1024
        if sys.platform == "darwin":
            import subprocess

            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                check=True,
            )
            return int(out.stdout.strip())
    except Exception:  # noqa: BLE001, S110 -- best-effort, non-critical diagnostic; failure modes vary by platform (ctypes, I/O, subprocess), "unknown" is a fine fallback for any of them
        pass
    return None


#: platform.processor() only returns a generic family/model/stepping string
#: on Windows, not the marketing CPU name -- set by the user for the
#: specific machine these benchmarks run on, since that can't be reliably
#: auto-detected. Falls back to an auto-generated label (OS + core count +
#: RAM) on any other machine. Update or clear this if the benchmarks move
#: to different hardware.
MACHINE_LABEL_OVERRIDE = (
    "Windows 11 laptop, Intel Core i7-11390H (4 cores / 8 threads, 3.4 GHz), 16 GB RAM"
)


def _env_info():
    ram = _ram_bytes()
    ram_gb = round(ram / (1024**3), 1) if ram else "unknown"
    cores = os.cpu_count()
    os_name = {"win32": "Windows", "darwin": "macOS", "linux": "Linux"}.get(
        sys.platform, sys.platform
    )
    machine_label = (
        MACHINE_LABEL_OVERRIDE
        or f"{os_name} machine, {cores} logical cores, {ram_gb}GB RAM"
    )
    networkx_version = nx.__version__
    if "rc" in networkx_version or "dev" in networkx_version:
        networkx_version += (
            " (development / release-candidate build, not a stable release)"
        )
    return {
        "machine_label": machine_label,
        "platform": platform.platform(),
        "cpu_model": platform.processor() or "unknown",
        "cpu_cores_logical": cores,
        "ram_gb": ram_gb,
        "python": sys.version.split()[0],
        "networkx": networkx_version,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _regenerate_plots_and_summary(
    stageb_rows,
    stagec_rows,
    env_info,
    elapsed,
    stable_rows,
    f5_edgescan_rows=None,
    hard_rows=None,
):
    f5_edgescan_rows = f5_edgescan_rows or []
    hard_rows = hard_rows if hard_rows is not None else _load_hard_rows()
    hard_rows = _apply_f5_idle_rerun(hard_rows)
    # Folded into the general stageb_rows used by P1/P2/P3 so the new
    # F5_chains_edgescan family shows up there too (more data, same plots);
    # P10 below is specific to it.
    stageb_rows_all = stageb_rows + f5_edgescan_rows

    print("\nGenerating plots...")
    plot_p1_iterations_vs_n(stageb_rows_all, PLOTDIR / "P1_iterations_vs_n.png")
    plot_p2_ops_per_iteration_over_m(
        stageb_rows_all, PLOTDIR / "P2_ops_per_iteration_over_m.png"
    )
    plot_p3_total_ops_over_bound(
        stageb_rows_all, PLOTDIR / "P3_total_ops_over_bound.png"
    )
    plot_p4_wallclock_loglog(stagec_rows, PLOTDIR / "P4_wallclock_loglog.png")
    plot_p5_normalized_time(stagec_rows, PLOTDIR / "P5_normalized_time.png")
    plot_p6_speedup(stagec_rows, PLOTDIR / "P6_speedup.png", greedy_init=True)
    plot_p6_speedup(
        stagec_rows, PLOTDIR / "P6b_speedup_greedy_off.png", greedy_init=False
    )
    n_plots = 7
    if hard_rows:
        plot_p8_hard_families(hard_rows, PLOTDIR / "P8_hard_families_vs_edmonds.png")
        plot_p9_speedup_all_families(
            stagec_rows, hard_rows, PLOTDIR / "P9_speedup_all_families.png"
        )
        n_plots += 2
    if f5_edgescan_rows:
        plot_p10_f5_edgescan(f5_edgescan_rows, PLOTDIR / "P10_f5_edgescan.png")
        n_plots += 1
    print(f"Saved {n_plots} plots to {PLOTDIR}")

    summary_path = write_summary(
        stageb_rows_all,
        stagec_rows,
        env_info,
        elapsed,
        OUTDIR / "SUMMARY.md",
        stable_rows=stable_rows,
        f5_edgescan_rows=f5_edgescan_rows,
        hard_rows=hard_rows,
    )
    print(f"Saved {summary_path}")


def _prior_elapsed_seconds():
    """Read the 'total run time' back out of the existing SUMMARY.md, for
    supplementary measurements (--stable-check, --f5-edgescan) that are not
    themselves a fresh --quick/--full collection and should not overwrite
    that number with their own, much shorter, run time.
    """
    import re

    old_summary_path = OUTDIR / "SUMMARY.md"
    if old_summary_path.exists():
        m = re.search(
            r"total run time\*\*: ([\d.]+)s",
            old_summary_path.read_text(encoding="utf-8"),
        )
        if m:
            return float(m.group(1))
    return float("nan")


def _load_supplementary_rows():
    stable_rows = (
        load_stable_check_csv()
        if (OUTDIR / "stageC_stable_check.csv").exists()
        else None
    )
    f5_edgescan_rows = load_f5_edgescan_csv() or None
    return stable_rows, f5_edgescan_rows


# F5 edge-scan points whose hard-families timings are replaced by the separate
# idle re-run (results/f5_idle_remeasure.csv: Gabow edge-scan x5, Edmonds x3,
# one process, no other benchmark running). The --hard-families timings at
# this size were taken while other work was running (Gabow 62-204s on
# identical work) and had a single Edmonds sample.
F5_IDLE_RERUN_NPARAMS = (20000,)
F5_IDLE_RERUN_CSV = "f5_idle_remeasure.csv"


def _apply_f5_idle_rerun(hard_rows):
    """Replace the hard-families F5 edge-scan Gabow rows and the Edmonds rows
    at F5_IDLE_RERUN_NPARAMS with the idle re-run's rows (marked
    `idle_rerun=True`). Returns hard_rows unchanged if the CSV is absent.
    """
    path = OUTDIR / F5_IDLE_RERUN_CSV
    if not path.exists():
        return hard_rows
    with open(path, newline="") as f:
        idle = [
            r for r in csv.DictReader(f) if int(r["n_param"]) in F5_IDLE_RERUN_NPARAMS
        ]
    if not idle:
        return hard_rows
    family = "F5_chains_edgescan_hard"
    replaced = {(int(r["n_param"]), r["algorithm"]) for r in idle}

    def is_replaced(r):
        if r["family"] != family:
            return False
        if r["algorithm"] == "gabow" and r["variant"] == "edgescan":
            return (r["n_param"], "gabow_edgescan") in replaced
        if r["algorithm"] == "edmonds":
            return (r["n_param"], "edmonds") in replaced
        return False

    out = [r for r in hard_rows if not is_replaced(r)]
    for r in idle:
        algo = "gabow" if r["algorithm"] == "gabow_edgescan" else "edmonds"
        out.append(
            {
                "family": family,
                "n": int(r["n"]),
                "n_param": int(r["n_param"]),
                "m": int(r["m"]),
                "seed": int(r["rep"]),
                "algorithm": algo,
                "variant": "edgescan" if algo == "gabow" else "none",
                "runtime_sec": float(r["runtime_sec"]),
                "iterations": int(r["iterations"]) if r["iterations"] else None,
                "idle_rerun": True,
            }
        )
    return out


def _load_hard_rows():
    if not (OUTDIR / "stageC_hardfamilies.csv").exists():
        return []
    return _load_rows_csv(
        "stageC_hardfamilies.csv",
        int_fields=["n", "n_param", "m", "seed", "iterations"],
        float_fields=["runtime_sec"],
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--full", action="store_true")
    p.add_argument(
        "--stable-check",
        action="store_true",
        help="Run ONLY the isolated, repeated F1 Edmonds-vs-Gabow check at "
        "n=10000 (run this in a fresh, otherwise-idle process), then "
        "regenerate plots/SUMMARY.md from the existing stageB_raw.csv / "
        "stageC_raw.csv on disk (does not re-run --quick/--full collection).",
    )
    p.add_argument(
        "--f5-edgescan",
        action="store_true",
        help="Run ONLY the F5 edge-scan-greedy-start experiment (sizes "
        f"{F5_EDGESCAN_SIZES}), then regenerate plots/SUMMARY.md from the "
        "existing stageB_raw.csv / stageC_raw.csv on disk (does not "
        "re-run --quick/--full collection).",
    )
    p.add_argument(
        "--hard-families",
        action="store_true",
        help="Run ONLY the F3-F6-vs-Edmonds hard-families comparison "
        "(stageC_hardfamilies.csv, resumable -- skips combos already "
        "in the CSV), using --full's sizes, then regenerate plots/"
        "SUMMARY.md.",
    )
    args = p.parse_args(argv)

    if args.stable_check or args.f5_edgescan or args.hard_families:
        env_info = _env_info()
        print("Environment:", env_info)
        t0 = time.perf_counter()
        if args.stable_check:
            run_f1_stable_check()
        if args.f5_edgescan:
            run_f5_edgescan()
        if args.hard_families:
            full_sizes = {
                "F3": [50, 200, 800, 1600],
                "F4": [11, 41, 161, 321],
                "F5": [10000, 20000, 40000],
                "F6": [1000, 10000, 100000],
            }
            run_stage_c_hard(full_sizes, n_seeds=5)
        print(f"\nRun time: {time.perf_counter() - t0:.1f}s")
        stageb_rows = load_stageb_csv()
        stagec_rows = load_stagec_csv()
        stable_rows, f5_edgescan_rows = _load_supplementary_rows()
        _regenerate_plots_and_summary(
            stageb_rows,
            stagec_rows,
            env_info,
            _prior_elapsed_seconds(),
            stable_rows,
            f5_edgescan_rows,
        )
        return

    if not args.quick and not args.full:
        p.error(
            "pass --quick, --full, --stable-check, --f5-edgescan, or --hard-families"
        )

    if args.quick:
        sizes = {
            "F1": [50, 200, 800],
            "F2": [50, 150, 400],
            "F3": [40, 200, 800],
            "F4": [11, 41, 161],
            "F5": [500, 1000, 2000],
            "F6": [100, 1000, 5000],
        }
        n_seeds = 2
        f5_edgescan_sizes = (500, 1000, 2000)
    else:
        sizes = {
            "F1": [1000, 5000, 20000, 100000],
            "F2": [50, 100, 200, 400, 800],
            "F3": [50, 200, 800, 1600],
            "F4": [11, 41, 161, 321],
            "F5": [10000, 20000, 40000],
            "F6": [1000, 10000, 100000],
        }
        n_seeds = 5
        f5_edgescan_sizes = F5_EDGESCAN_SIZES

    env_info = _env_info()
    print("Environment:", env_info)

    t0 = time.perf_counter()
    stageb_rows, _stageb_families = run_stage_b(sizes, n_seeds)
    stagec_rows = run_stage_c(sizes, n_seeds)
    run_f5_edgescan(sizes=f5_edgescan_sizes)
    hard_rows = run_stage_c_hard(sizes, n_seeds)
    elapsed = time.perf_counter() - t0

    stable_rows, f5_edgescan_rows = _load_supplementary_rows()

    _regenerate_plots_and_summary(
        stageb_rows,
        stagec_rows,
        env_info,
        elapsed,
        stable_rows,
        f5_edgescan_rows,
        hard_rows,
    )
    print(f"\nTotal run time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
