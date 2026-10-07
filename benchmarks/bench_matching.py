"""Runtime benchmark comparing two maximum-cardinality matching
implementations that already live in networkx.algorithms.matching:

  * ``max_cardinality_matching_gabow`` -- Gabow's phase-based method,
    claimed O(sqrt(n) * m) (Theorem 5.1 of "The Weighted Matching Approach
    to Maximum Cardinality Matching", H.N. Gabow, Fundamenta Informaticae
    154 (2017)).
  * ``max_weight_matching(G, maxcardinality=True)`` -- the existing
    Galil/Edmonds primal-dual blossom implementation, documented as
    O(number_of_nodes ** 3) in its own docstring.

    NOTE: max_weight_matching is a *weighted*-matching routine. Here it is
    run on uniformly unweighted random graphs purely to solve the same
    maximum-cardinality-matching problem Gabow's method solves -- so this
    is a comparison of two different ALGORITHMS solving the same problem
    via different formulations, not a strictly apples-to-apples
    weighted-vs-unweighted benchmark. It is, however, the closest existing
    baseline for maximum cardinality matching in NetworkX.

This script does not modify either function; it only measures them.

Usage:
    python benchmarks/bench_matching.py
    python benchmarks/bench_matching.py --quick            # fast smoke test
    python benchmarks/bench_matching.py --sizes 50 100 200 --densities 0.1 0.5
    python benchmarks/bench_matching.py --no-loglog-inset
    python benchmarks/bench_matching.py --show             # also plt.show()

Outputs (under --outdir, default benchmarks/results/):
    runtime_density_10.png, runtime_density_25.png, ... (one per density)
    raw_results.csv     (density, n, m, seed, algorithm, runtime_sec)
    summary.csv         (density, n, algorithm, mean, median, std, repetitions)
and a final printed summary table of fitted vs. theoretical growth
exponents, with the standard error of each fitted slope.

Statistics
----------
Repetitions (independent random graphs, i.e. independent seeds) are
adaptive by size by default -- more repetitions for the cheap small sizes,
fewer for the expensive large ones -- see REPEATS_SCHEDULE / repeats_for_n().
For each (density, n) we record the mean, median, and std of the runtime
across repetitions; the plots use the MEAN with +/-1 std error bars, and the
log-log regression (for the fitted slope) is fit on the means. A call whose
single-shot timing comes in under FAST_CALL_THRESHOLD_SEC is re-measured as
a batch of several calls on the same (already-copied) graph object, with the
batch's total time divided by the call count, so that timer/function-call
overhead doesn't dominate the measurement of very fast (mostly Gabow) runs.
"""

import argparse
import csv
import statistics
import time
import warnings
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import networkx as nx

DENSITIES = [0.10, 0.25, 0.50, 0.75, 1.00]
SIZES = [50, 100, 150, 200, 250, 300, 400, 500, 800]

# Adaptive repetitions (independent random graphs/seeds) per graph size:
# more repetitions where each trial is cheap, fewer where it's expensive.
# Expressed as (max_n_inclusive, repeats) bands, checked in order.
REPEATS_SCHEDULE = [
    (150, 30),  # n <= 150
    (300, 15),  # 150 < n <= 300
    (float("inf"), 10),  # n > 300
]

QUICK_SIZES = [20, 40, 60]
QUICK_DENSITIES = [0.25, 0.75]
QUICK_REPEATS = 2

# Below this single-call elapsed time, re-measure as a batch (see
# _time_call_adaptive) so timer/call overhead doesn't dominate.
FAST_CALL_THRESHOLD_SEC = 0.02
MAX_BATCH_CALLS = 500

# key -> (display label, callable(G) -> matching)
ALGORITHMS = {
    "gabow": (
        "Gabow (ours)",
        lambda G: nx.max_cardinality_matching_gabow(G),
    ),
    "edmonds": (
        "NetworkX Edmonds",
        lambda G: nx.max_weight_matching(G, maxcardinality=True),
    ),
}

# Theoretical log-log growth exponent in n, at *fixed* density.
# Gabow: O(sqrt(n) * m); m = density * n*(n-1)/2 = Theta(n**2) at fixed
# density, so sqrt(n) * n**2 = n**2.5 regardless of which density level.
# Edmonds (max_weight_matching docstring): O(n**3), stated purely in terms
# of n, so also density-independent.
THEORETICAL_SLOPE = {"gabow": 2.5, "edmonds": 3.0}


def repeats_for_n(n, schedule=REPEATS_SCHEDULE):
    for max_n, reps in schedule:
        if n <= max_n:
            return reps
    return schedule[-1][1]


def edges_for_density(n, density):
    """Number of edges for a gnm_random_graph(n, m) at the given density,
    where density 1.0 means the complete graph K_n.
    """
    max_edges = n * (n - 1) // 2
    return round(density * max_edges)


def _verify_non_mutating():
    """One-time sanity check that neither algorithm mutates its input
    graph. The batch-timing path below reuses a single copied graph object
    across many calls purely as a performance optimization (avoiding
    per-call G.copy() overhead from dominating fast Gabow measurements);
    that optimization is only valid if this holds.
    """
    G = nx.gnm_random_graph(30, 90, seed=0)
    before = (G.number_of_nodes(), G.number_of_edges(), tuple(sorted(G.edges())))
    for _key, (_label, fn) in ALGORITHMS.items():
        fn(G)
        after = (G.number_of_nodes(), G.number_of_edges(), tuple(sorted(G.edges())))
        if after != before:
            raise RuntimeError(
                f"{_label} appears to mutate its input graph; the "
                "batch-timing optimization in this script assumes it "
                "doesn't. Disable batching or fix the reuse."
            )


def _time_call_adaptive(fn, G):
    """Time one call to fn(G). If it's faster than FAST_CALL_THRESHOLD_SEC,
    re-measure as a batch of repeated calls on the same graph object
    (divided by call count) so timer/call overhead doesn't dominate.
    """
    Gc = G.copy()
    t0 = time.perf_counter()
    result = fn(Gc)
    t1 = time.perf_counter()
    elapsed = t1 - t0
    if elapsed >= FAST_CALL_THRESHOLD_SEC:
        return elapsed, result

    est_calls = int(FAST_CALL_THRESHOLD_SEC / max(elapsed, 1e-7)) + 1
    batch = min(MAX_BATCH_CALLS, max(2, est_calls))
    t0 = time.perf_counter()
    for _ in range(batch):
        fn(Gc)
    t1 = time.perf_counter()
    return (t1 - t0) / batch, result


def run_experiments(sizes, densities, repeats_override=None, base_seed=0):
    """Returns a list of row dicts: density, n, m, seed, algorithm,
    runtime_sec -- the raw data backing both the CSV and the plots.

    `repeats_override`, if given, is used as a fixed repetition count for
    every size (matching the old behavior / the --quick config); otherwise
    repeats_for_n() picks an adaptive count per size.
    """
    _verify_non_mutating()

    rows = []
    total = len(sizes) * len(densities)
    done = 0
    for density in densities:
        for n in sizes:
            done += 1
            m = edges_for_density(n, density)
            repeats = (
                repeats_override if repeats_override is not None else repeats_for_n(n)
            )
            print(
                f"[{done}/{total}] density={density:.2f} n={n} m={m} "
                f"repeats={repeats} ...",
                flush=True,
            )

            # One untimed warm-up call per algorithm, on a same-size
            # throwaway graph, before any timed repetitions.
            warm_G = nx.gnm_random_graph(n, m, seed=base_seed - 1)
            for _key, (_label, fn) in ALGORITHMS.items():
                fn(warm_G.copy())

            for rep in range(repeats):
                seed = base_seed + rep
                G = nx.gnm_random_graph(n, m, seed=seed)

                per_algo_size = {}
                for key, (_label, fn) in ALGORITHMS.items():
                    elapsed, result = _time_call_adaptive(fn, G)
                    per_algo_size[key] = len(result)
                    rows.append(
                        {
                            "density": density,
                            "n": n,
                            "m": m,
                            "seed": seed,
                            "algorithm": key,
                            "runtime_sec": elapsed,
                        }
                    )

                sizes_found = set(per_algo_size.values())
                if len(sizes_found) != 1:
                    raise AssertionError(
                        "Matching cardinality mismatch at density="
                        f"{density}, n={n}, seed={seed}: {per_algo_size} "
                        "-- the two algorithms disagree on the maximum "
                        "matching size, which should never happen."
                    )
    return rows


def write_csv(rows, outdir):
    path = Path(outdir) / "raw_results.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["density", "n", "m", "seed", "algorithm", "runtime_sec"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def _stats(rows, density, algorithm, n):
    vals = [
        r["runtime_sec"]
        for r in rows
        if r["density"] == density and r["algorithm"] == algorithm and r["n"] == n
    ]
    return {
        "mean": statistics.mean(vals),
        "median": statistics.median(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "repetitions": len(vals),
    }


def write_summary_csv(rows, densities, sizes, outdir):
    path = Path(outdir) / "summary.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "density",
                "n",
                "algorithm",
                "mean",
                "median",
                "std",
                "repetitions",
            ],
        )
        writer.writeheader()
        for density in densities:
            for n in sizes:
                for key in ALGORITHMS:
                    s = _stats(rows, density, key, n)
                    writer.writerow({"density": density, "n": n, "algorithm": key, **s})
    return path


def _aggregate_means(rows, density, algorithm, sizes):
    """mean/std runtime per n, for one (density, algorithm), in the same
    order as `sizes`.
    """
    means, stds = [], []
    for n in sizes:
        s = _stats(rows, density, algorithm, n)
        means.append(s["mean"])
        stds.append(s["std"])
    return means, stds


def _fit_loglog_slope(sizes, means):
    """Fit log(t) = k*log(n) + c via ordinary least squares on the MEAN
    runtimes. Returns (slope, intercept, r2, se_slope), where se_slope is
    the standard error of the fitted slope:

        se_slope = sqrt( (SS_res / (n - 2)) / Sxx ),  Sxx = sum((x-xbar)**2)

    the usual simple-linear-regression formula. Guards against numpy's
    RankWarning on tiny/degenerate inputs, which would otherwise become a
    hard error under this repo's filterwarnings=["error"] pytest setting.
    """
    log_n = np.log(np.asarray(sizes, dtype=float))
    log_t = np.log(np.asarray(means, dtype=float))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.exceptions.RankWarning)
        slope, intercept = np.polyfit(log_n, log_t, 1)
    fitted = slope * log_n + intercept
    resid = log_t - fitted
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((log_t - np.mean(log_t)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    n_pts = len(log_n)
    sxx = float(np.sum((log_n - np.mean(log_n)) ** 2))
    if n_pts > 2 and sxx > 0:
        s2 = ss_res / (n_pts - 2)
        se_slope = float(np.sqrt(s2 / sxx))
    else:
        se_slope = float("nan")

    return float(slope), float(intercept), r2, se_slope


def make_plot(rows, density, sizes, outdir, loglog_inset, show=False):
    fig_summary = {}

    if loglog_inset:
        fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(12, 5))
    else:
        fig, ax_lin = plt.subplots(1, 1, figsize=(7, 5))
        ax_log = None

    colors = {"gabow": "tab:blue", "edmonds": "tab:orange"}

    for key, (label, _fn) in ALGORITHMS.items():
        means, stds = _aggregate_means(rows, density, key, sizes)
        slope, intercept, r2, se_slope = _fit_loglog_slope(sizes, means)
        fig_summary[key] = {"slope": slope, "r2": r2, "se_slope": se_slope}

        legend_label = f"{label}, slope = {slope:.2f}"
        ax_lin.errorbar(
            sizes,
            means,
            yerr=stds,
            marker="o",
            color=colors[key],
            capsize=3,
            label=legend_label,
        )

        if ax_log is not None:
            log_n = np.log(np.asarray(sizes, dtype=float))
            log_t = np.log(np.asarray(means, dtype=float))
            ax_log.plot(log_n, log_t, "o", color=colors[key], label=label)
            fitted = slope * log_n + intercept
            ax_log.plot(
                log_n,
                fitted,
                "--",
                color=colors[key],
                label=f"{label} fit (k={slope:.2f}+/-{se_slope:.2f}, R2={r2:.3f})",
            )

    pct = round(density * 100)
    ax_lin.set_title(f"Running Time - Density {pct}%")
    ax_lin.set_xlabel("Graph Size [Nodes]")
    ax_lin.set_ylabel("Running Time [sec]")
    ax_lin.grid(True, axis="y", color="lightgray", linewidth=0.8)
    ax_lin.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1)

    if ax_log is not None:
        # Dashed theoretical reference lines, anchored to the first
        # algorithm's data point so they're visually comparable in scale.
        log_n_full = np.log(np.asarray(sizes, dtype=float))
        for key in ALGORITHMS:
            means, _stds = _aggregate_means(rows, density, key, sizes)
            log_t = np.log(np.asarray(means, dtype=float))
            theo_slope = THEORETICAL_SLOPE[key]
            theo_intercept = log_t[0] - theo_slope * log_n_full[0]
            theo_line = theo_slope * log_n_full + theo_intercept
            ax_log.plot(
                log_n_full,
                theo_line,
                ":",
                color=colors[key],
                alpha=0.6,
                label=f"{ALGORITHMS[key][0]} theory (k={theo_slope})",
            )
        ax_log.set_title(f"Log-log fit - Density {pct}%")
        ax_log.set_xlabel("log(Graph Size)")
        ax_log.set_ylabel("log(Running Time)")
        ax_log.grid(True, color="lightgray", linewidth=0.8)
        ax_log.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1, fontsize=8)

    fig.tight_layout()
    outpath = Path(outdir) / f"runtime_density_{pct}.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")

    if show:
        try:
            plt.show()
        except Exception as exc:  # pragma: no cover - depends on display/backend
            print(f"(--show requested but plt.show() failed: {exc})")

    plt.close(fig)
    return outpath, fig_summary


def print_summary_table(all_summaries, densities, total_runtime_sec):
    print()
    print("=" * 90)
    print("SUMMARY: empirical vs. theoretical log-log growth exponent (runtime ~ n^k)")
    print("=" * 90)
    header = (
        f"{'density':>8} {'algorithm':>16} {'measured k':>11} {'se(k)':>8} "
        f"{'R^2':>7} {'theoretical k':>14}"
    )
    print(header)
    print("-" * len(header))
    for density in densities:
        for key, (label, _fn) in ALGORITHMS.items():
            s = all_summaries[density][key]
            print(
                f"{density:>8.2f} {label:>16} {s['slope']:>11.2f} "
                f"{s['se_slope']:>8.2f} {s['r2']:>7.3f} "
                f"{THEORETICAL_SLOPE[key]:>14.2f}"
            )
    print("-" * len(header))
    print(
        "Note: at fixed density, m = Theta(n**2), so the theoretical exponent\n"
        "is the SAME across all density levels for a given algorithm -- only\n"
        "the constant factor (and hence absolute runtime) changes with density."
    )
    print()
    print(f"Total benchmark run time: {total_runtime_sec:.1f}s")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", type=int, nargs="+", default=None)
    p.add_argument("--densities", type=float, nargs="+", default=None)
    p.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="fixed repetition count for every size, overriding the adaptive "
        "schedule (more repetitions for small n, fewer for large n)",
    )
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument(
        "--loglog-inset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="add a log-log side panel with fitted + theoretical reference lines",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="also call plt.show() for each figure (in addition to saving it)",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="tiny smoke-test configuration (overridden by explicit --sizes/--densities/--repeats)",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.quick:
        sizes = args.sizes or QUICK_SIZES
        densities = args.densities or QUICK_DENSITIES
        repeats_override = args.repeats if args.repeats is not None else QUICK_REPEATS
    else:
        sizes = args.sizes or SIZES
        densities = args.densities or DENSITIES
        repeats_override = args.repeats  # None => adaptive schedule

    outdir = Path(args.outdir) if args.outdir else Path(__file__).with_name("results")
    outdir.mkdir(parents=True, exist_ok=True)

    schedule_desc = (
        f"fixed repeats={repeats_override}"
        if repeats_override is not None
        else f"adaptive repeats schedule={REPEATS_SCHEDULE}"
    )
    print(f"sizes={sizes} densities={densities} {schedule_desc} outdir={outdir}")

    run_t0 = time.perf_counter()
    rows = run_experiments(sizes, densities, repeats_override=repeats_override)
    run_t1 = time.perf_counter()

    csv_path = write_csv(rows, outdir)
    print(f"\nSaved raw results to {csv_path}")
    summary_csv_path = write_summary_csv(rows, densities, sizes, outdir)
    print(f"Saved summary statistics to {summary_csv_path}")

    all_summaries = {}
    for density in densities:
        outpath, fig_summary = make_plot(
            rows,
            density,
            sizes,
            outdir,
            loglog_inset=args.loglog_inset,
            show=args.show,
        )
        all_summaries[density] = fig_summary
        print(f"Saved plot to {outpath}")

    total_runtime_sec = run_t1 - run_t0
    print_summary_table(all_summaries, densities, total_runtime_sec)
    return rows, all_summaries


if __name__ == "__main__":
    main()
