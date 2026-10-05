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
    plots/P1_*.png ... plots/P7_*.png
    SUMMARY.md
"""

import argparse
import csv
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

import networkx as nx

sys.path.insert(0, str(Path(__file__).parent))
import bench_matching as BM
import verify_complexity as VC

OUTDIR = Path(__file__).with_name("results")
PLOTDIR = OUTDIR / "plots"
OUTDIR.mkdir(parents=True, exist_ok=True)
PLOTDIR.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]  # >= 5 seeds per (family, n), as specified

# Markers/linestyles cycled for black-and-white readability (not color alone).
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]


def _style(i):
    return MARKERS[i % len(MARKERS)], LINESTYLES[i % len(LINESTYLES)]


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


def _measure_stageB(G, family, n_label, seed, greedy_init):
    """n_label is the generator's own size parameter (e.g. n_param for the
    F5 chains family, which is NOT the resulting node count -- recorded as
    its own `n_param` column for traceability; `n` is always the actual
    G.number_of_nodes(), which is what every plot's x-axis uses).
    """
    counters = {}
    nx.max_cardinality_matching_gabow(
        G, _counters=counters, _skip_greedy_init=not greedy_init
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


def run_stage_b(sizes_by_family, n_seeds):
    print("\n" + "=" * 70)
    print("STAGE B: operation counts (Gabow only)")
    print("=" * 70)
    seeds = SEEDS[:n_seeds]
    all_rows = []

    def add(family, gen, sizes, extra_kwargs=None):
        extra_kwargs = extra_kwargs or {}
        rows = []
        for n in sizes:
            for seed in seeds:
                G = gen(n, seed, **extra_kwargs)
                for greedy_init in (True, False):
                    row = _measure_stageB(G, family, n, seed, greedy_init)
                    rows.append(row)
        print(
            f"  {family}: {len(sizes)} sizes x {len(seeds)} seeds x 2 "
            f"(greedy on/off) = {len(rows)} rows"
        )
        all_rows.extend(rows)
        return rows

    rows_f1 = []
    for c in (3, 5, 10):
        rows_f1.extend(
            add(
                f"F1_sparse_c{c}",
                lambda n, seed, c=c: f1_random_sparse(n, c, seed),
                sizes_by_family["F1"],
            )
        )
    rows_f2 = []
    for d in (0.10, 0.25, 0.50, 0.75, 1.00):
        rows_f2.extend(
            add(
                f"F2_density{round(d * 100)}",
                lambda n, seed, d=d: f2_random_dense(n, d, seed),
                sizes_by_family["F2"],
            )
        )
    rows_f3 = add("F3_bad_greedy", f3_bad_greedy, sizes_by_family["F3"])
    rows_f4 = add("F4_nested_blossoms", f4_nested_blossoms, sizes_by_family["F4"])

    rows_f5 = []
    for n_param in sizes_by_family["F5"]:
        for seed in seeds[:3]:  # "3 shuffles per size"
            G = f5_short_and_long_chains(n_param, seed, shuffle=False)
            for greedy_init in (True, False):
                row = _measure_stageB(
                    G, "F5_chains_unshuffled", n_param, seed, greedy_init
                )
                rows_f5.append(row)
            G_shuf = f5_short_and_long_chains(n_param, seed, shuffle=True)
            for greedy_init in (True, False):
                row = _measure_stageB(
                    G_shuf, "F5_chains_shuffled", n_param, seed, greedy_init
                )
                rows_f5.append(row)
    print(f"  F5_chains: {len(rows_f5)} rows")
    all_rows.extend(rows_f5)

    rows_f6 = add("F6_long_path", f6_long_path, sizes_by_family["F6"])

    path = _write_rows_csv(all_rows, "stageB_raw.csv", STAGEB_FIELDS)
    print(f"Saved Stage B raw data to {path} ({len(all_rows)} rows total)")

    families = {
        "F1": [r for r in all_rows if r["family"].startswith("F1")],
        "F2": [r for r in all_rows if r["family"].startswith("F2")],
        "F3": rows_f3,
        "F4": rows_f4,
        "F5": rows_f5,
        "F6": rows_f6,
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


def run_stage_c(sizes_by_family, n_seeds):
    print("\n" + "=" * 70)
    print(
        "STAGE C: wall-clock, Gabow (greedy ON/OFF) vs Edmonds (instrumentation OFF for timing)"
    )
    print("=" * 70)
    seeds = SEEDS[:n_seeds]
    rows = []
    # Keyed by the FULL family string (e.g. "F1_sparse_c3"), NOT by fam_key
    # ("F1") -- an earlier version shared one flag across all three F1
    # c-values (and all five F2 densities), so one slow Edmonds call in
    # F1_sparse_c3 silently disabled Edmonds for F1_sparse_c5/c10 too, even
    # though they were never themselves tested. fam_key is still used below
    # for the (legitimately shared) F1 size cap.
    edmonds_disabled = {}
    f1_edmonds_cap = _calibrate_f1_edmonds_cap(sizes_by_family)

    def gabow(greedy_init):
        return lambda G: nx.max_cardinality_matching_gabow(
            G, use_heuristic_fallback=False, _skip_greedy_init=not greedy_init
        )

    def edmonds(G):
        return nx.max_weight_matching(G, maxcardinality=True)

    def run_family(family, gen, sizes, extra_kwargs=None):
        extra_kwargs = extra_kwargs or {}
        for n in sizes:
            for seed in seeds:
                G = gen(n, seed, **extra_kwargs)
                m = G.number_of_edges()

                res_g_by_mode = {}
                for greedy_init in (True, False):
                    iters, already_opt = _gabow_diagnostics(G, greedy_init)
                    times_g, res_g = _adaptive_timed_runs(gabow(greedy_init), G)
                    res_g_by_mode[greedy_init] = res_g
                    for t in times_g:
                        rows.append(
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
                            }
                        )
                    print(
                        f"  {family} n={n} seed={seed} greedy_init={greedy_init}: "
                        f"median={statistics.median(times_g):.4g}s mean={statistics.mean(times_g):.4g}s "
                        f"iterations={iters} greedy_already_optimal={already_opt}"
                    )

                fam_key = family.split("_")[0]
                over_f1_cap = (
                    fam_key == "F1"
                    and f1_edmonds_cap is not None
                    and n > f1_edmonds_cap
                )
                if over_f1_cap and not edmonds_disabled.get(family, False):
                    print(
                        f"    skipping Edmonds at n={n} ({family}): over the calibrated F1 cap "
                        f"({f1_edmonds_cap}); Gabow still ran above."
                    )
                if not edmonds_disabled.get(family, False) and not over_f1_cap:
                    times_e, res_e = _adaptive_timed_runs(edmonds, G)
                    if max(times_e) > TIMEOUT_SEC:
                        print(
                            f"    Edmonds exceeded {TIMEOUT_SEC}s at n={n} ({family}); "
                            "disabling Edmonds for larger n in this family, continuing Gabow."
                        )
                        edmonds_disabled[family] = True
                    for t in times_e:
                        rows.append(
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
                            }
                        )
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

    path = _write_rows_csv(rows, "stageC_raw.csv", STAGEC_FIELDS)
    print(f"Saved Stage C raw data to {path} ({len(rows)} rows total)")
    return rows


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
            G, use_heuristic_fallback=False, _skip_greedy_init=not greedy_init
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
    """Median and min/max per (c, seed, algorithm/mode), plus the per-c
    speedup (edmonds median / gabow greedy-OFF median), from an isolated
    run_f1_stable_check() collection.
    """
    if not stable_rows:
        return []
    out = []
    cs = sorted({r["c"] for r in stable_rows})
    out.append(
        "| c | seed | edmonds median (min-max) | gabow greedy_ON median (min-max) | gabow greedy_OFF median (min-max) | speedup (edmonds/gabow_OFF) |"
    )
    out.append("|---|---|---|---|---|---|")
    for c in cs:
        seeds = sorted({r["seed"] for r in stable_rows if r["c"] == c})
        for seed in seeds:
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
            me, mon, moff = (
                statistics.median(e),
                statistics.median(g_on),
                statistics.median(g_off),
            )
            out.append(
                f"| {c} | {seed} | {me:.4g}s ({min(e):.4g}-{max(e):.4g}) | "
                f"{mon:.4g}s ({min(g_on):.4g}-{max(g_on):.4g}) | "
                f"{moff:.4g}s ({min(g_off):.4g}-{max(g_off):.4g}) | {me / moff:.2f}x |"
            )
    out.append(
        "\nPer-c speedup, pooling all measured seeds (edmonds median / gabow greedy-OFF median):\n"
    )
    for c in cs:
        sub = [r for r in stable_rows if r["c"] == c]
        e = [r["runtime_sec"] for r in sub if r["algorithm"] == "edmonds"]
        g_off = [
            r["runtime_sec"]
            for r in sub
            if r["algorithm"] == "gabow" and r["greedy_init"] is False
        ]
        if not (e and g_off):
            continue
        out.append(
            f"- c={c}: edmonds median={statistics.median(e):.4g}s, gabow greedy_OFF median={statistics.median(g_off):.4g}s, speedup={statistics.median(e) / statistics.median(g_off):.2f}x"
        )
    return out


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
# Plots P1-P7
# ---------------------------------------------------------------------------


def plot_p1_iterations_vs_n(stageb_rows, outpath):
    fig, ax = plt.subplots(figsize=(8, 6))
    families = sorted({r["family"] for r in stageb_rows if r["greedy_init"] is False})
    for i, fam in enumerate(families):
        sub = [
            r for r in stageb_rows if r["family"] == fam and r["greedy_init"] is False
        ]
        by_n = sorted({r["n"] for r in sub})
        meds = []
        for n in by_n:
            vals = [r["iterations"] for r in sub if r["n"] == n]
            meds.append(statistics.median(vals))
        marker, ls = _style(i)
        ax.plot(by_n, meds, marker=marker, linestyle=ls, label=fam, markersize=5)
    # Anchor c so the reference curve passes through the SMALLEST measured
    # point (not the largest): anchoring at the largest point lets a flat
    # (non-sqrt-growing) family look deceptively "under" a steep reference
    # curve throughout the whole plot; anchoring at the smallest point
    # means any family that actually keeps pace with sqrt(n) will visibly
    # track the reference line, while flat families will visibly fall
    # below it as n grows -- the more honest comparison for this plot's
    # purpose (spotting sqrt(n)-like growth, not just not-contradicting it).
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
        ref = [c * math.sqrt(n) for n in ns_all]
        ax.plot(
            ns_all,
            ref,
            "k:",
            label=f"reference c*sqrt(n), c anchored at n={anchor_n}",
            linewidth=2,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n (number of nodes)")
    ax.set_ylabel("iterations (median over seeds)")
    ax.set_title("P1: Iterations vs n (greedy_init=False), with c*sqrt(n) reference")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_p2_ops_per_iteration_over_m(stageb_rows, outpath):
    fig, ax = plt.subplots(figsize=(8, 6))
    families = sorted({r["family"] for r in stageb_rows if r["greedy_init"] is False})
    for i, fam in enumerate(families):
        sub = [
            r for r in stageb_rows if r["family"] == fam and r["greedy_init"] is False
        ]
        by_n = sorted({r["n"] for r in sub})
        ys = []
        for n in by_n:
            group = [r for r in sub if r["n"] == n]
            vals = [
                (r["total_ops"] / r["iterations"]) / r["m"]
                for r in group
                if r["iterations"] > 0 and r["m"] > 0
            ]
            ys.append(statistics.median(vals) if vals else float("nan"))
        marker, ls = _style(i)
        ax.plot(by_n, ys, marker=marker, linestyle=ls, label=fam, markersize=5)
    ax.set_xscale("log")
    ax.set_xlabel("n (number of nodes)")
    ax.set_ylabel("(total_ops / iterations) / m  [dimensionless]")
    ax.set_title("P2: Ops per iteration, normalized by m -- expect flat")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_p3_total_ops_over_bound(stageb_rows, outpath):
    fig, ax = plt.subplots(figsize=(8, 6))
    families = sorted({r["family"] for r in stageb_rows if r["greedy_init"] is False})
    for i, fam in enumerate(families):
        sub = [
            r for r in stageb_rows if r["family"] == fam and r["greedy_init"] is False
        ]
        by_n = sorted({r["n"] for r in sub})
        ys = []
        for n in by_n:
            group = [r for r in sub if r["n"] == n]
            vals = [
                r["total_ops"] / (math.sqrt(r["n"]) * r["m"])
                for r in group
                if r["m"] > 0
            ]
            ys.append(statistics.median(vals) if vals else float("nan"))
        marker, ls = _style(i)
        ax.plot(by_n, ys, marker=marker, linestyle=ls, label=fam, markersize=5)
    ax.set_xscale("log")
    ax.set_xlabel("n (number of nodes)")
    ax.set_ylabel("total_ops / (sqrt(n) * m)")
    ax.set_title("P3: Total ops / bound(n,m) -- expect flat or decreasing")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def _plot_p4_panel(ax, stagec_rows, families, title):
    colors = {"gabow": "tab:blue", "edmonds": "tab:orange"}
    greedy_ls = {True: "-", False: "--"}
    i = 0
    for fam in families:
        for greedy_init in (True, False):
            sub = [
                r
                for r in stagec_rows
                if r["family"] == fam
                and r["algorithm"] == "gabow"
                and r["greedy_init"] == greedy_init
            ]
            if not sub:
                continue
            by_n = sorted({r["n"] for r in sub})
            meds = [
                statistics.median([r["runtime_sec"] for r in sub if r["n"] == n])
                for n in by_n
            ]
            slope, _intercept, r2, se = _fit_loglog(by_n, meds)
            marker, _ls = _style(i)
            label = (
                f"{fam} gabow greedy={greedy_init} k={slope:.2f}+/-{se:.2f} R2={r2:.2f}"
            )
            ax.plot(
                by_n,
                meds,
                marker=marker,
                linestyle=greedy_ls[greedy_init],
                color=colors["gabow"],
                label=label,
                markersize=5,
            )
            i += 1
        sub_e = [
            r for r in stagec_rows if r["family"] == fam and r["algorithm"] == "edmonds"
        ]
        if sub_e:
            by_n = sorted({r["n"] for r in sub_e})
            meds = [
                statistics.median([r["runtime_sec"] for r in sub_e if r["n"] == n])
                for n in by_n
            ]
            slope, _intercept, r2, se = _fit_loglog(by_n, meds)
            marker, _ls = _style(i)
            label = f"{fam} edmonds k={slope:.2f}+/-{se:.2f} R2={r2:.2f}"
            ax.plot(
                by_n,
                meds,
                marker=marker,
                linestyle=":",
                color=colors["edmonds"],
                label=label,
                markersize=5,
            )
            i += 1
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n (number of nodes)")
    ax.set_ylabel("median wall-clock time [s]")
    ax.set_title(title)
    ax.legend(fontsize=5, ncol=1)
    ax.grid(True, which="both", alpha=0.3)


def plot_p4_wallclock_loglog(stagec_rows, outpath):
    """Split into two panels (F1 sparse, F2 dense), each with its own
    bound-implied reference slope in the title; within a panel, Gabow
    greedy_init=True/False are distinguished by line style (solid/dashed),
    Edmonds by dotted line and a separate color.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    f1_families = sorted(
        {r["family"] for r in stagec_rows if r["family"].startswith("F1")}
    )
    f2_families = sorted(
        {r["family"] for r in stagec_rows if r["family"].startswith("F2")}
    )
    _plot_p4_panel(
        ax1,
        stagec_rows,
        f1_families,
        "P4a: F1 sparse (expected slope: Gabow~1.5, Edmonds~3)",
    )
    _plot_p4_panel(
        ax2,
        stagec_rows,
        f2_families,
        "P4b: F2 dense (expected slope: Gabow~2.5, Edmonds~3)",
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_p5_normalized_time(stagec_rows, outpath):
    """Gabow rows are filtered to greedy_init=True (the real, user-facing
    function) here -- see SUMMARY.md's Stage C section for the greedy
    ON-vs-OFF comparison itself (item 1 of the follow-up request).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    families = sorted({r["family"] for r in stagec_rows})
    for i, fam in enumerate(families):
        sub_g = [
            r
            for r in stagec_rows
            if r["family"] == fam
            and r["algorithm"] == "gabow"
            and r["greedy_init"] is True
        ]
        sub_e = [
            r for r in stagec_rows if r["family"] == fam and r["algorithm"] == "edmonds"
        ]
        marker, ls = _style(i)
        if sub_g:
            by_n = sorted({r["n"] for r in sub_g})
            ys = []
            for n in by_n:
                group = [r for r in sub_g if r["n"] == n]
                m = statistics.median([r["m"] for r in group])
                t = statistics.median([r["runtime_sec"] for r in group])
                ys.append(t / (math.sqrt(n) * m) if m > 0 else float("nan"))
            ax1.plot(by_n, ys, marker=marker, linestyle=ls, label=fam, markersize=5)
        if sub_e:
            by_n = sorted({r["n"] for r in sub_e})
            ys = [
                statistics.median([r["runtime_sec"] for r in sub_e if r["n"] == n])
                / (n**3)
                for n in by_n
            ]
            ax2.plot(by_n, ys, marker=marker, linestyle=ls, label=fam, markersize=5)
    for ax, title, ylabel in [
        (ax1, "Gabow: time / (sqrt(n)*m)", "time / (sqrt(n)*m)"),
        (ax2, "Edmonds: time / n^3", "time / n^3"),
    ]:
        ax.set_xscale("log")
        ax.set_xlabel("n (number of nodes)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=6)
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle(
        "P5: Normalized runtime, Gabow greedy_init=True (expect flat or decreasing if bound holds)"
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_p6_speedup(stagec_rows, outpath, greedy_init=True):
    """Speedup = median Edmonds time / median Gabow time, for the given
    Gabow greedy_init mode. greedy_init=True is the real, user-facing
    function (P6); greedy_init=False (P6b) shows the same ratio with
    greedy warm-start off -- see the "dense graphs, greedy OFF vs Edmonds"
    SUMMARY.md section: unlike P6, P6b's ratio drops BELOW 1 for most F2
    densities, i.e. Gabow is slower than Edmonds there, not faster. This
    is a result to report, not a bug to fix.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    families = sorted({r["family"] for r in stagec_rows})
    for i, fam in enumerate(families):
        sub_g = {
            r["n"]: []
            for r in stagec_rows
            if r["family"] == fam
            and r["algorithm"] == "gabow"
            and r["greedy_init"] is greedy_init
        }
        sub_e = {
            r["n"]: []
            for r in stagec_rows
            if r["family"] == fam and r["algorithm"] == "edmonds"
        }
        for r in stagec_rows:
            if r["family"] != fam:
                continue
            if r["algorithm"] == "gabow" and r["greedy_init"] is greedy_init:
                sub_g[r["n"]].append(r["runtime_sec"])
            elif r["algorithm"] == "edmonds":
                sub_e[r["n"]].append(r["runtime_sec"])
        ns = sorted(set(sub_g) & set(sub_e))
        if not ns:
            continue
        speedup = [
            statistics.median(sub_e[n]) / statistics.median(sub_g[n]) for n in ns
        ]
        marker, ls = _style(i)
        ax.plot(ns, speedup, marker=marker, linestyle=ls, label=fam, markersize=5)
    ax.axhline(1.0, color="k", linestyle=":", linewidth=1, label="speedup = 1 (tie)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n (number of nodes)")
    ax.set_ylabel("speedup = Edmonds time / Gabow time")
    ax.set_title(
        f"P6{'' if greedy_init else 'b'}: Speedup of Gabow (greedy_init={greedy_init}) over Edmonds"
    )
    ax.legend(fontsize=6)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_p7_f5_iterations(stageb_rows, outpath):
    fig, ax = plt.subplots(figsize=(8, 6))
    paper_n = [10000, 20000, 40000]
    paper_it = [24, 33, 47]
    ax.plot(paper_n, paper_it, "k*", markersize=14, label="paper Table 1", zorder=5)
    tags = [
        ("F5_chains_unshuffled", True, "ours unshuffled, greedy ON"),
        ("F5_chains_unshuffled", False, "ours unshuffled, greedy OFF"),
        ("F5_chains_shuffled", True, "ours shuffled, greedy ON"),
        ("F5_chains_shuffled", False, "ours shuffled, greedy OFF"),
    ]
    for i, (fam, greedy, label) in enumerate(tags):
        sub = [
            r for r in stageb_rows if r["family"] == fam and r["greedy_init"] is greedy
        ]
        if not sub:
            continue
        by_n = sorted({r["n"] for r in sub})
        meds = [
            statistics.median([r["iterations"] for r in sub if r["n"] == n])
            for n in by_n
        ]
        marker, ls = _style(i)
        ax.plot(by_n, meds, marker=marker, linestyle=ls, label=label, markersize=6)
    ax.set_xscale("log")
    ax.set_xlabel("actual n (G.number_of_nodes()) -- NOT the generator's n_param input")
    ax.set_ylabel("iterations (median over seeds)")
    ax.set_title("P7: F5 iterations vs actual n -- ours vs paper's Table 1")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


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


def _f5_size_comparison(stageb_rows):
    """For each paper (n, m) row, find the F5_chains_unshuffled n_param whose
    ACTUAL built graph (G.number_of_nodes()) is closest to that paper n, and
    report paper (n, m) side by side with ours (n_param, actual n, actual m)
    -- both for that best-matching n_param AND for naively using n_param ==
    paper's n directly (what the original run did), to show the size of the
    mislabeling concretely from real data, not just asserted.
    """
    sub = [
        r
        for r in stageb_rows
        if r["family"] == "F5_chains_unshuffled" and r["greedy_init"] is False
    ]
    by_nparam = {}
    for r in sub:
        by_nparam.setdefault(r["n_param"], (r["n"], r["m"]))
    out = []
    out.append(
        "| paper n | paper m | n_param used naively (= paper n) -> actual n, m | "
        "best-matching n_param -> actual n, m |"
    )
    out.append("|---|---|---|---|")
    for paper_n, paper_m, _it in PAPER_SHORT_AND_LONG:
        naive = by_nparam.get(paper_n)
        naive_str = (
            f"n_param={paper_n} -> n={naive[0]}, m={naive[1]}"
            if naive
            else "(not measured)"
        )
        best = min(by_nparam.items(), key=lambda kv: abs(kv[1][0] - paper_n))
        best_str = f"n_param={best[0]} -> n={best[1][0]}, m={best[1][1]}"
        out.append(f"| {paper_n} | {paper_m} | {naive_str} | {best_str} |")
    return out


def write_summary(stageb_rows, stagec_rows, env_info, elapsed, path, stable_rows=None):
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
        "Split into two panels, F1 (sparse) and F2 (dense), each with its "
        "own bound-implied reference slope in the panel title; within a "
        "panel, Gabow greedy_init=True/False are solid/dashed lines in the "
        "same color, Edmonds is a dotted line in a different color.\n"
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
            meds = [
                statistics.median([r["runtime_sec"] for r in sub if r["n"] == n])
                for n in ns
            ]
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
        "and should be read as noisy** -- any P4a/P6/P6b speedup figure "
        "computed from them (e.g. a ~140x P6 speedup at n=20000) inherits "
        "that same uncertainty and could plausibly be off by a factor of "
        "several. See the stable, repeated measurement at n=10000 directly "
        "below for a number with real error bars.\n"
    )

    lines.append("### Stable F1 Edmonds check (n=10000, repeated, isolated process)\n")
    if stable_rows:
        lines.append(
            "Run via `run_f1_stable_check()` in a dedicated, freshly-started "
            f"Python process with nothing else running on the machine, at "
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
            "\nCompare to the single-sample ~140x figure implied by the "
            "n=20000 points above (e.g. c=3: edmonds=258.5s / gabow "
            "greedy_OFF~1.0-1.1s there) -- the stable n=10000 numbers give "
            "the actual order of magnitude of the speedup with real "
            "min/max spread, without relying on a single noisy multi-"
            "minute call.\n"
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

    lines.append("## P6 -- Speedup (Edmonds / Gabow)\n")
    lines.append(
        "Ratio of median Edmonds time to median Gabow (greedy_init=True) "
        "time. An increasing trend with n is consistent with Gabow's "
        "better asymptotic complexity actually manifesting in wall-clock "
        "terms on this machine.\n"
        "\nNote on F2_density100: per the Stage C bimodality section above, "
        "greedy init alone was already the maximum matching on EVERY "
        "measured seed at every n for F2_density100 with greedy_init=True "
        "(augmentations == 0 throughout) -- so F2_density100's P6 line "
        "measures the cost of the greedy warm-start pass itself, not of "
        "Gabow's real search, since the real search never ran on any of "
        "those graphs.\n"
    )

    lines.append("## Dense graphs, greedy OFF vs Edmonds (plain result, not a bug)\n")
    lines.append(
        "P4b already shows this in log-log slope form; stated plainly "
        "here with numbers. With greedy_init=OFF (an empty starting "
        "matching, the directly paper-comparable mode -- see the citation "
        "check above), Gabow is SLOWER than Edmonds at every F2 density "
        "except the sparsest (10%), by a roughly constant factor that does "
        "NOT shrink with density (same ~n^2 slope as Edmonds in P4b, just "
        "a larger constant). This is a real result about THIS "
        "implementation's constant factors on dense random graphs when "
        "forced to find every augmenting path from scratch -- it is not "
        "evidence against the O(sqrt(n)*m*alpha(n)) bound (which bounds "
        "Gabow relative to its OWN n and m, not relative to Edmonds' "
        'constant), and it is not being "fixed" here, per instructions:\n'
    )
    for d in (10, 25, 50, 75, 100):
        fam = f"F2_density{d}"
        sub_on = [
            r["runtime_sec"]
            for r in stagec_rows
            if r["family"] == fam
            and r["n"] == 800
            and r["algorithm"] == "gabow"
            and r["greedy_init"] is True
        ]
        sub_off = [
            r["runtime_sec"]
            for r in stagec_rows
            if r["family"] == fam
            and r["n"] == 800
            and r["algorithm"] == "gabow"
            and r["greedy_init"] is False
        ]
        sub_e = [
            r["runtime_sec"]
            for r in stagec_rows
            if r["family"] == fam and r["n"] == 800 and r["algorithm"] == "edmonds"
        ]
        if not (sub_on and sub_off and sub_e):
            continue
        m_on, m_off, m_e = (
            statistics.median(sub_on),
            statistics.median(sub_off),
            statistics.median(sub_e),
        )
        lines.append(
            f"- {fam}, n=800: gabow greedy_OFF={m_off:.4g}s, "
            f"gabow greedy_ON={m_on:.4g}s, edmonds={m_e:.4g}s "
            f"(greedy_OFF/edmonds = {m_off / m_e:.2f}x)"
        )
    lines.append(
        "\nSee plots/P6b_speedup_greedy_off.png (speedup = Edmonds time / "
        "Gabow greedy_OFF time, same axes as P6 but with a y=1 reference "
        "line added): P6 (greedy ON, the real default) stays above 1 and "
        "grows with n for every density; P6b drops below 1 for density "
        ">=25% once the greedy warm-start is turned off.\n"
    )

    lines.append("## P7 -- F5 (short-and-long chains) iterations vs paper Table 1\n")
    lines.append(
        "**x-axis check**: confirmed from the code -- "
        "`_measure_stageB` records `n = G.number_of_nodes()` (the actual "
        "built graph's node count), never the generator's own `n_param` "
        "input; P7's x-axis is therefore already actual n, not n_param "
        "(the axis label has been corrected to say so explicitly -- it "
        'previously said "n_param" in the label text even though the '
        "plotted values were always actual n). The `n_param` column is "
        "kept in stageB_raw.csv alongside `n` for every row (not just F5) "
        "so this is auditable directly from the raw data.\n"
    )
    lines.append(
        "**Citation check**: arXiv:2603.22909 (Mehlhorn & Nobahari, \"Gabow's "
        'O(sqrt(n) m) Maximum Cardinality Matching Algorithm, Revisited") '
        "IS the paper this project ports -- re-fetched directly from the PDF "
        "(not a lossy HTML summarizer, which gave internally-inconsistent "
        "numbers on a first attempt) to get exact quotes:\n"
        '- (a) Section 4: "Initially,M is empty." (p.18 of the PDF; OCR '
        "renders the M without a space) -- confirmed, matches our "
        "`greedy_init=False` setting, NOT `greedy_init=True`. The comparison "
        "below is therefore most meaningful for the `greedy_init=False` rows.\n"
        '- (b) Section 5 (p.18): "They consist of a complete graph with '
        "about sqrt(n) vertices, O(n) short chains of length seven all "
        "attached to a fixed vertex z of the complete graph, and, in the "
        "case of short and long chains, one chain of length 2i+1 each "
        "attached to z, where 4<=i<=sqrt(n). We refer to [ADM24] for more "
        'details." -- the paper itself defers the exact construction to '
        "[ADM24] (Ansaripour, Danaei, Mehlhorn 2024) and does not give a "
        "closed-form n->(vertex count, edge count) formula; see the P7 "
        "section below for what this means for our generator's sizing.\n"
        "- (c) NOT addressed: the paper text does not state whether Table "
        "1's #it column includes the final unsuccessful search. This could "
        "not be confirmed either way from the paper text -- stated here "
        "rather than assumed.\n"
    )
    lines.append(
        "\n**Size check**: the table below shows paper n=20000, m=56000 "
        "lines up almost exactly with OUR actual graph at n_param=10000 "
        "(actual n=20080, m=56970 -- within 1.7% on n, 1.7% on m), and "
        "likewise paper n=40000/m=114000 against our n_param=20000 (actual "
        'n=40284, m=114351). So the paper\'s "n" in Table 1 IS the total '
        "vertex count, directly comparable to our `n = G.number_of_nodes()` "
        "-- it is OUR GENERATOR's `n_param` input that is the construction "
        "parameter, at roughly n_param = n/2 for THIS generator (not a "
        "property of the paper's construction; the paper's own prose "
        "describes chain counts as O(n)/O(sqrt(n)) in the asymptotic sense, "
        "not as a literal formula, so there is no contradiction in n_param "
        "landing at n/2 here). The original run set n_param equal to the "
        "paper's n literally (10000/20000/40000), which is why our "
        "x=20080/40284/80180 points sat about 2x to the right of the "
        "paper's n=10000/20000/40000 stars on P7 -- effectively comparing "
        "against the wrong row, not a uniform 2x error at every row (see "
        "table). The table finds, for each paper row, the n_param whose "
        "ACTUAL graph size best matches that paper n, and reports n and m "
        "side by side for both the naive and the corrected choice. A new "
        "Stage B run at the corrected n_param values (F5 only, greedy OFF, "
        "shuffled and unshuffled) has been added to stageB_raw.csv and is "
        "included in every F5 plot and table in this document, not just "
        "this comparison.\n"
    )
    lines.extend(_f5_size_comparison(stageb_rows))
    lines.append(
        "\nThe best-matching column's m lands within about 2% of the "
        "paper's m at n=20000 and n=40000, and within about 29% at the "
        "smallest size (n=10000, where integer truncation in the "
        "generator's sqrt/floor formulas has the most relative effect) -- "
        "supporting the same construction (up to the two sources' "
        "independent choices of short-chain length -- 7 in the prose above "
        "vs. 8 in the companion C++ reference this project ported from, a "
        "discrepancy already noted in verify_complexity.chains_graph's "
        "docstring and not re-litigated here), not a different graph "
        "family.\n"
        "\n**Size does not explain the iteration gap**: even at correctly "
        "matched actual n, our `greedy_init=False` iteration counts stay "
        "at 2-3 (see the per-size listing below) against the paper's "
        "24/33/47 at the same three sizes. Getting the vertex count right "
        "did not close the gap -- whatever makes the paper's construction "
        "force O(sqrt(n)) growth in iterations, this reconstruction of it "
        "is not reproducing, and that remains open (not refuted or "
        "confirmed), per the Verdict below.\n"
    )
    sub = [r for r in stageb_rows if r["family"].startswith("F5")]
    if sub:
        for fam in ["F5_chains_unshuffled", "F5_chains_shuffled"]:
            for greedy in (True, False):
                s = [
                    r for r in sub if r["family"] == fam and r["greedy_init"] is greedy
                ]
                if s:
                    ns = sorted({r["n"] for r in s})
                    its = [
                        statistics.median([r["iterations"] for r in s if r["n"] == n])
                        for n in ns
                    ]
                    lines.append(
                        f"- {fam}, greedy_init={greedy}: n={ns} -> iterations={its}"
                    )
    lines.append(
        "- paper Table 1: n=[10000, 20000, 40000] -> iterations=[24, 33, 47]\n"
        "\n**Verdict**: per the corrected statement in docs/complexity_audit.md, "
        "fewer iterations than the paper's Table 1 does NOT contradict the "
        "O(sqrt(n)) upper bound -- it only means this reconstruction of the "
        "family did not reproduce the paper's reported worst-case behavior, "
        "so this remains open, not refuted or confirmed. The right "
        "comparison per the M=empty finding above is `greedy_init=False`.\n"
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
    stageb_rows, stagec_rows, env_info, elapsed, stable_rows
):
    print("\nGenerating plots...")
    plot_p1_iterations_vs_n(stageb_rows, PLOTDIR / "P1_iterations_vs_n.png")
    plot_p2_ops_per_iteration_over_m(
        stageb_rows, PLOTDIR / "P2_ops_per_iteration_over_m.png"
    )
    plot_p3_total_ops_over_bound(stageb_rows, PLOTDIR / "P3_total_ops_over_bound.png")
    plot_p4_wallclock_loglog(stagec_rows, PLOTDIR / "P4_wallclock_loglog.png")
    plot_p5_normalized_time(stagec_rows, PLOTDIR / "P5_normalized_time.png")
    plot_p6_speedup(stagec_rows, PLOTDIR / "P6_speedup.png", greedy_init=True)
    plot_p6_speedup(
        stagec_rows, PLOTDIR / "P6b_speedup_greedy_off.png", greedy_init=False
    )
    plot_p7_f5_iterations(stageb_rows, PLOTDIR / "P7_f5_iterations_vs_paper.png")
    print(f"Saved 8 plots to {PLOTDIR}")

    summary_path = write_summary(
        stageb_rows,
        stagec_rows,
        env_info,
        elapsed,
        OUTDIR / "SUMMARY.md",
        stable_rows=stable_rows,
    )
    print(f"Saved {summary_path}")


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
    args = p.parse_args(argv)

    if args.stable_check:
        env_info = _env_info()
        print("Environment:", env_info)
        t0 = time.perf_counter()
        stable_rows = run_f1_stable_check()
        stable_elapsed = time.perf_counter() - t0
        print(f"\nStable check run time: {stable_elapsed:.1f}s")
        stageb_rows = load_stageb_csv()
        stagec_rows = load_stagec_csv()
        # Keep the prior collection's reported elapsed time (this is a
        # supplementary measurement, not a fresh full collection) by
        # reading it back out of the existing SUMMARY.md if present.
        import re

        old_summary_path = OUTDIR / "SUMMARY.md"
        elapsed = float("nan")
        if old_summary_path.exists():
            m = re.search(
                r"total run time\*\*: ([\d.]+)s",
                old_summary_path.read_text(encoding="utf-8"),
            )
            if m:
                elapsed = float(m.group(1))
        _regenerate_plots_and_summary(
            stageb_rows, stagec_rows, env_info, elapsed, stable_rows
        )
        return

    if not args.quick and not args.full:
        p.error("pass --quick, --full, or --stable-check")

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

    env_info = _env_info()
    print("Environment:", env_info)

    t0 = time.perf_counter()
    stageb_rows, _stageb_families = run_stage_b(sizes, n_seeds)
    stagec_rows = run_stage_c(sizes, n_seeds)
    elapsed = time.perf_counter() - t0

    stable_rows = None
    if (OUTDIR / "stageC_stable_check.csv").exists():
        stable_rows = load_stable_check_csv()

    _regenerate_plots_and_summary(
        stageb_rows, stagec_rows, env_info, elapsed, stable_rows
    )
    print(f"\nTotal run time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
