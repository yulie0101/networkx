"""Generate plots from bigrun's own CSVs (results/bigrun_complexity.csv).
Same spirit as benchmarks/run_all.py's P1-P9, re-implemented here so this
package has no dependency on the rest of the repo.

Usage: python make_plots.py
"""

import csv
import math
import statistics
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
RESULTS = HERE / "results"
PLOTS = RESULTS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)


def load_rows():
    path = RESULTS / "bigrun_complexity.csv"
    if not path.exists():
        print("no bigrun_complexity.csv yet -- run bigrun.py first")
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fit_loglog(xs, ys):
    xs = [math.log(x) for x in xs]
    ys = [math.log(y) for y in ys if y > 0]
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan"), float("nan")
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx else float("nan")
    intercept = my - slope * mx
    return slope, intercept


def p1_iterations_vs_n(rows):
    fig, ax = plt.subplots(figsize=(9, 6.5))
    families = sorted({r["family"] for r in rows})
    for family in families:
        sub = [
            r
            for r in rows
            if r["family"] == family
            and r["algorithm"] == "gabow"
            and r["variant"] == "greedy_off"
        ]
        pts = sorted(
            {(int(r["n"]), int(r["iterations"])) for r in sub if r["iterations"]}
        )
        if not pts:
            continue
        xs = sorted({x for x, _ in pts})
        ys = [statistics.median([it for n, it in pts if n == x]) for x in xs]
        slope, _ = fit_loglog(xs, ys)
        ax.plot(xs, ys, marker="o", label=f"{family} (k={slope:.2f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("iterations (greedy off)")
    ax.set_title("P1: iterations vs n (expect slope ~0.5)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "P1_iterations_vs_n.png", dpi=150)
    plt.close(fig)


def p2_ops_over_bound(rows):
    fig, ax = plt.subplots(figsize=(9, 6.5))
    families = sorted({r["family"] for r in rows})
    for family in families:
        sub = [
            r
            for r in rows
            if r["family"] == family
            and r["algorithm"] == "gabow"
            and r["variant"] == "greedy_off"
            and r["total_ops"]
        ]
        ns = sorted({int(r["n"]) for r in sub})
        ys = []
        for n in ns:
            vals = [
                int(r["total_ops"]) / (math.sqrt(n) * int(r["m"]))
                for r in sub
                if int(r["n"]) == n
            ]
            ys.append(statistics.median(vals))
        if ns:
            ax.plot(ns, ys, marker="s", label=family)
    ax.set_xscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("total_ops / (sqrt(n) * m)")
    ax.set_title("P2: ops / bound (expect flat)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "P2_ops_over_bound.png", dpi=150)
    plt.close(fig)


def p3_wallclock(rows):
    families = sorted({r["family"] for r in rows})
    fig, axes = plt.subplots(1, len(families), figsize=(6 * len(families), 5.5))
    if len(families) == 1:
        axes = [axes]
    for ax, family in zip(axes, families):
        sub = [r for r in rows if r["family"] == family]
        for algo_variant in sorted({(r["algorithm"], r["variant"]) for r in sub}):
            algo, variant = algo_variant
            label = "edmonds" if algo == "edmonds" else variant
            pts = [
                r
                for r in sub
                if r["algorithm"] == algo
                and r["variant"] == variant
                and r["timed_out"] != "True"
            ]
            ns = sorted({int(r["n"]) for r in pts})
            ys = [
                statistics.median(
                    [float(r["runtime_sec"]) for r in pts if int(r["n"]) == n]
                )
                for n in ns
            ]
            if ns:
                ax.plot(ns, ys, marker="o", label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("n")
        ax.set_ylabel("median runtime (s)")
        ax.set_title(f"P3: {family}")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "P3_wallclock_per_family.png", dpi=150)
    plt.close(fig)


def p4_speedup(rows):
    fig, ax = plt.subplots(figsize=(9, 6.5))
    families = sorted({r["family"] for r in rows})
    for family in families:
        sub = [r for r in rows if r["family"] == family]
        ns = sorted({int(r["n"]) for r in sub})
        xs, ys = [], []
        for n in ns:
            g = [
                float(r["runtime_sec"])
                for r in sub
                if int(r["n"]) == n
                and r["algorithm"] == "gabow"
                and r["variant"] == "greedy_on"
            ]
            e = [
                float(r["runtime_sec"])
                for r in sub
                if int(r["n"]) == n
                and r["algorithm"] == "edmonds"
                and r["timed_out"] != "True"
            ]
            if g and e:
                xs.append(n)
                ys.append(statistics.median(e) / statistics.median(g))
        if xs:
            ax.plot(xs, ys, marker="o", label=family)
    ax.axhline(1.0, color="black", linewidth=1, label="y=1")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("speedup (edmonds / gabow greedy-on)")
    ax.set_title("P4: speedup vs n, all families")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "P4_speedup_all_families.png", dpi=150)
    plt.close(fig)


def main():
    rows = load_rows()
    if not rows:
        return
    p1_iterations_vs_n(rows)
    p2_ops_over_bound(rows)
    p3_wallclock(rows)
    p4_speedup(rows)
    print(f"Saved 4 plots to {PLOTS}")


if __name__ == "__main__":
    main()
