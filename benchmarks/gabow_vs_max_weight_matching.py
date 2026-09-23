"""Runtime comparison: max_cardinality_matching_gabow vs. NetworkX's
existing max_weight_matching(maxcardinality=True) run on uniform (unweighted)
graphs -- i.e. both solving the same problem, maximum cardinality matching,
via two different algorithms (Gabow's O(sqrt(n) * m) phase-based method vs.
the O(n**3) Galil/Edmonds primal-dual method).

This is a standalone script, not an asv benchmark and not collected by
pytest:

    python benchmarks/gabow_vs_max_weight_matching.py

It times both functions on the *same* sparse random graphs (so the
comparison is apples-to-apples) across doubling sizes, runs
max_cardinality_matching_gabow with use_heuristic_fallback=False (we want to
compare Gabow's algorithm itself, not the constant-factor fallback), and
saves a log-log runtime plot to gabow_vs_max_weight_matching.png next to
this script. Since max_weight_matching is cubic it becomes impractically
slow well before Gabow's algorithm does, so once a single max_weight_matching
run exceeds MWM_TIME_BUDGET seconds it is skipped for larger sizes while
max_cardinality_matching_gabow keeps going.
"""

import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import networkx as nx

SIZES = [50, 100, 200, 400, 800, 1600, 3200, 6400]
DENSITY = 3  # m = DENSITY * n edges per graph
REPEATS = 3
MWM_TIME_BUDGET = 20.0  # seconds; stop growing max_weight_matching past this


def _timed_best_of(fn, G, repeats=REPEATS):
    best = math.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(G)
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    sizes_used = []
    gabow_times = []
    mwm_times = []  # None once max_weight_matching has been disabled
    mwm_enabled = True

    print(f"{'n':>6}{'m':>8}{'gabow (s)':>14}{'max_weight_matching (s)':>26}")
    for n in SIZES:
        m = DENSITY * n
        G = nx.gnm_random_graph(n, m, seed=n)

        gabow_t = _timed_best_of(
            lambda g: nx.max_cardinality_matching_gabow(
                g, use_heuristic_fallback=False
            ),
            G,
        )

        if mwm_enabled:
            mwm_t = _timed_best_of(
                lambda g: nx.max_weight_matching(g, maxcardinality=True), G, repeats=1
            )
            if mwm_t > MWM_TIME_BUDGET:
                mwm_enabled = False
        else:
            mwm_t = None

        sizes_used.append(n)
        gabow_times.append(gabow_t)
        mwm_times.append(mwm_t)
        mwm_str = "skipped (too slow)" if mwm_t is None else f"{mwm_t:.4f}"
        print(f"{n:>6}{m:>8}{gabow_t:>14.4f}{mwm_str:>26}")

    # -- plot ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        sizes_used,
        gabow_times,
        marker="o",
        label="max_cardinality_matching_gabow (O(sqrt(n)*m))",
    )
    mwm_x = [n for n, t in zip(sizes_used, mwm_times) if t is not None]
    mwm_y = [t for t in mwm_times if t is not None]
    ax.plot(
        mwm_x,
        mwm_y,
        marker="s",
        label="max_weight_matching(maxcardinality=True) (O(n**3))",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"number of nodes n (sparse random graph, m = {DENSITY}*n edges)")
    ax.set_ylabel("runtime in seconds (best of 3 runs; mwm best of 1 at larger n)")
    ax.set_title("Maximum cardinality matching: Gabow vs. Galil/Edmonds")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    outpath = Path(__file__).with_name("gabow_vs_max_weight_matching.png")
    fig.savefig(outpath, dpi=150)
    print(f"\nSaved plot to {outpath}")

    # -- summary of relative speed at a couple of representative sizes ------
    print()
    for n, gt, mt in zip(sizes_used, gabow_times, mwm_times):
        if mt is None:
            continue
        faster, slower = ("gabow", "max_weight_matching") if gt < mt else (
            "max_weight_matching",
            "gabow",
        )
        factor = max(gt, mt) / min(gt, mt)
        print(
            f"n={n}: {faster} is faster than {slower} by ~{factor:.1f}x "
            f"(gabow={gt:.4f}s, max_weight_matching={mt:.4f}s)"
        )


if __name__ == "__main__":
    main()
