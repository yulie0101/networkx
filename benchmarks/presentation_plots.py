"""Slide-ready charts for the presentation, built from the existing results.

Reads only the CSVs that ``run_all.py`` already wrote to ``results/`` (no new
runs) and writes one PNG per chart plus ``captions.md`` to
``results/presentation/``. Re-run it after ``run_all.py`` to refresh the
charts::

    python benchmarks/presentation_plots.py

Aggregation is the same as in ``run_all.py``/SUMMARY.md: for each graph the
median of its timed runs, then the median over graphs; speedups are the
median of per-graph speedups. Operation counts are medians over seeds.

Style: one question per chart, linear axes, at most three lines with
markers, large fonts, light horizontal grid, legend below the plot, 16:9 at
200 dpi. Gabow (greedy ON) is dark blue, Gabow (greedy OFF) dark blue
dashed, Edmonds (``max_weight_matching``) orange.
"""

import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import run_all as R
from matplotlib.ticker import FuncFormatter

OUT = R.OUTDIR / "presentation"

BLUE = "#1f3b73"
ORANGE = "#f28e2b"
GREY = "#8c8c8c"
#: Colors for lines that compare graph types, not algorithms.
TYPE_COLORS = ["#2a9d8f", "#7b2cbf", "#555555"]

STYLE = {
    "gabow_on": {"color": BLUE, "ls": "-", "marker": "o", "label": "Gabow (greedy ON)"},
    "gabow_off": {
        "color": BLUE,
        "ls": "--",
        "marker": "s",
        "label": "Gabow (greedy OFF)",
    },
    "gabow_cpp": {
        "color": BLUE,
        "ls": ":",
        "marker": "D",
        "label": "Gabow (authors' greedy start)",
    },
    "edmonds": {
        "color": ORANGE,
        "ls": "-",
        "marker": "^",
        "label": "Edmonds (max_weight_matching)",
    },
}

SPARSE = [(3, "F1_sparse_c3"), (5, "F1_sparse_c5"), (10, "F1_sparse_c10")]
DENSE = [(10, "F2_density10"), (25, "F2_density25"), (50, "F2_density50")]
DENSE += [(75, "F2_density75"), (100, "F2_density100")]
HARD = [
    ("F3_bad_greedy", "Bad greedy start (paths of 4 nodes)", "hard_bad_greedy"),
    ("F4_nested_blossoms", "Nested blossoms (chained triangles)", "hard_nested"),
    ("F5_chains_edgescan_hard", "Short and long chains", "hard_chains"),
    ("F6_long_path", "One long path", "hard_long_path"),
]

#: Broom family (two search trees joined by K_{k,k}, started from its own
#: matching), measured once before and after the Phase 1 climb fix
#: (68d14aa52). Source: docs/complexity_audit.md Sec. 9; the compiled C++
#: reference walks the same numbers of steps as the code before the fix.
BROOM = {
    "L": [40, 80, 160],
    "n": [322, 642, 1282],
    "m": [1920, 7040, 26880],
    "steps_before": [65600, 518400, 4121600],
    "steps_after": [0, 0, 0],
}
BROOM_TIME_L160 = (6.90, 0.34)  # seconds before / after, L = 160

captions = []


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _fmt_int(v, _pos=None):
    return f"{v:,.0f}"


def _new_chart(title, subtitle=None):
    fig, ax = plt.subplots(figsize=(12, 6.75))
    fig.suptitle(title, fontsize=20, fontweight="bold", y=0.97)
    if subtitle:
        ax.set_title(subtitle, fontsize=14, color="#444444", pad=12)
    ax.grid(axis="y", color="#dddddd", linewidth=1)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=14)
    return fig, ax


def _finish(fig, ax, name, xlabel, ylabel, legend=True, int_x=True):
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    if int_x:
        ax.xaxis.set_major_formatter(FuncFormatter(_fmt_int))
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(len(labels), 3),
            fontsize=14,
            frameon=False,
        )
        fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)


def _line(ax, xs, ys, style, **kw):
    st = dict(STYLE[style]) if isinstance(style, str) else dict(style)
    ax.plot(
        xs,
        ys,
        color=st["color"],
        linestyle=st["ls"],
        marker=st["marker"],
        linewidth=3,
        markersize=9,
        label=kw.get("label", st["label"]),
    )


def _join_types(labels):
    dense = [x.split(", ")[1] for x in labels if x.startswith("Dense, ")]
    rest = [x for x in labels if not x.startswith("Dense, ")]
    if dense:
        rest.insert(0, f"dense graphs ({', '.join(dense)})")
    return ", ".join(rest)


def _caption(name, title, takeaway, numbers):
    captions.append((name, title, takeaway, numbers))


def _num(v):
    if v is None:
        return "-"
    if v >= 100:
        return f"{v:,.0f}"
    if v >= 1:
        return f"{v:.2f}"
    return f"{v:.3g}"


def _sec(v):
    return "-" if v is None else f"{_num(v)} s"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def _time_series(rows, family, key):
    """{n: point time} for one variant of one family."""
    if key == "edmonds":
        sel = [r for r in rows if r["algorithm"] == "edmonds"]
    elif key in ("gabow_on", "gabow_off"):
        on = key == "gabow_on"
        sel = [
            r
            for r in rows
            if r["algorithm"] == "gabow"
            and (
                r.get("greedy_init") is on
                or r.get("variant") == ("greedy_on" if on else "greedy_off")
            )
        ]
    else:  # gabow_cpp
        sel = [r for r in rows if r.get("variant") == "edgescan"]
    sel = [r for r in sel if r["family"] == family]
    out = {}
    for n in sorted({r["n"] for r in sel}):
        out[n] = R._point_time([r for r in sel if r["n"] == n])
    return out


def _speedup(rows, family, key, n):
    e = [r for r in rows if r["family"] == family and r["n"] == n]
    g = [
        r
        for r in e
        if r["algorithm"] == "gabow"
        and (
            r.get("greedy_init") is (key == "gabow_on")
            or r.get("variant") == ("greedy_on" if key == "gabow_on" else "greedy_off")
        )
    ]
    e = [r for r in e if r["algorithm"] == "edmonds"]
    sp = R._point_speedup(e, g)
    return sp[0] if sp else None


def _ops_by_n(rows, family, f):
    """{n: median over seeds of f(row)}, greedy OFF (empty start)."""
    sel = [r for r in rows if r["family"] == family and r["greedy_init"] is False]
    return {
        n: statistics.median(f(r) for r in sel if r["n"] == n)
        for n in sorted({r["n"] for r in sel})
    }


# --------------------------------------------------------------------------
# A. Comparison with Edmonds
# --------------------------------------------------------------------------


def time_chart(rows, family, title, name, variants):
    fig, ax = _new_chart(title)
    series = {k: _time_series(rows, family, k) for k in variants}
    for k in variants:
        s = series[k]
        if s:
            _line(ax, list(s), list(s.values()), k)
    _finish(fig, ax, name, "Graph size [nodes]", "Running time [sec]")
    ns = sorted(set().union(*series.values()))
    nums = []
    for k in variants:
        pts = ", ".join(f"{n:,}: {_sec(v)}" for n, v in series[k].items())
        nums.append(f"{STYLE[k]['label']}: {pts}")
    big = max(n for n in series["edmonds"]) if series["edmonds"] else None
    if big is not None:
        parts = [
            f"{STYLE[k]['label'].split(' (')[0] if k == 'edmonds' else STYLE[k]['label']}"
            f" {_sec(series[k].get(big))}"
            for k in variants
            if series[k].get(big) is not None
        ]
        takeaway = f"At {big:,} nodes: " + "; ".join(parts) + "."
    else:
        takeaway = f"Sizes {ns[0]:,}-{ns[-1]:,} nodes."
    _caption(name, title, takeaway, nums)
    return series


def comparison_charts(stagec, hard):
    for d, fam in DENSE:
        s = time_chart(
            stagec,
            fam,
            f"Running Time - Density {d}%",
            f"A1_dense_{d}",
            ["gabow_on", "gabow_off", "edmonds"],
        )
        n = max(s["edmonds"])
        on, off = (
            _speedup(stagec, fam, "gabow_on", n),
            _speedup(stagec, fam, "gabow_off", n),
        )
        captions[-1] = captions[-1][:2] + (
            captions[-1][2]
            + f" Gabow is {on:.2g}x as fast as Edmonds with the greedy start and"
            f" {off:.2g}x without it (median of per-graph speedups).",
            captions[-1][3],
        )
    for c, fam in SPARSE:
        s = time_chart(
            stagec,
            fam,
            f"Running Time - Sparse graphs (m = {c}n)",
            f"A2_sparse_{c}n",
            ["gabow_on", "gabow_off", "edmonds"],
        )
        n = max(s["edmonds"])
        on = _speedup(stagec, fam, "gabow_on", n)
        off = _speedup(stagec, fam, "gabow_off", n)
        captions[-1] = captions[-1][:2] + (
            captions[-1][2]
            + f" Gabow is {on:,.0f}x faster with the greedy start and {off:,.0f}x"
            f" without it; Edmonds was stopped after {n:,} nodes (too slow).",
            captions[-1][3],
        )
    for fam, title, name in HARD:
        variants = ["gabow_on", "gabow_off", "edmonds"]
        if fam == "F5_chains_edgescan_hard":
            variants = ["gabow_on", "gabow_cpp", "edmonds"]
        s = time_chart(hard, fam, f"Running Time - {title}", f"A3_{name}", variants)
        n = max(s["edmonds"])
        on = _speedup(hard, fam, "gabow_on", n)
        extra = f" Gabow is {on:,.0f}x faster than Edmonds with the greedy start"
        if fam == "F5_chains_edgescan_hard":
            cpp = s["gabow_cpp"][n]
            extra += (
                f" and {s['edmonds'][n] / cpp:.1f}x faster from the authors' start,"
                " which forces about sqrt(n) iterations."
            )
        else:
            off = _speedup(hard, fam, "gabow_off", n)
            extra += f" and {off:,.0f}x without it."
        captions[-1] = captions[-1][:2] + (captions[-1][2] + extra, captions[-1][3])


def speedup_bars(stagec, hard):
    groups = []  # (label, family, rows, n)
    for c, fam in SPARSE:
        groups.append((f"Sparse, m = {c}n", fam, stagec))
    for d, fam in DENSE:
        groups.append((f"Dense, {d}%", fam, stagec))
    for fam, title, _ in HARD:
        groups.append((title.split(" (")[0], fam, hard))
    bars = []
    for label, fam, rows in groups:
        ns = {
            r["n"] for r in rows if r["family"] == fam and r["algorithm"] == "edmonds"
        }
        n = max(ns)
        bars.append(
            (
                f"{label}\n({n:,} nodes)",
                _speedup(rows, fam, "gabow_off", n),
                _speedup(rows, fam, "gabow_on", n),
            )
        )
    title = "How many times faster than Edmonds"
    fig, ax = plt.subplots(figsize=(12, 6.75))
    fig.suptitle(title, fontsize=20, fontweight="bold", y=0.97)
    ax.set_title(
        "At the largest size where both were run; above 1 = Gabow is faster",
        fontsize=14,
        color="#444444",
        pad=12,
    )
    ys = list(range(len(bars)))[::-1]
    h = 0.38
    off = [b[1] for b in bars]
    on = [b[2] for b in bars]
    ax.barh(
        [y + h / 2 for y in ys],
        off,
        h,
        color="white",
        edgecolor=BLUE,
        hatch="//",
        linewidth=1.5,
        label="Gabow (greedy OFF)",
    )
    ax.barh([y - h / 2 for y in ys], on, h, color=BLUE, label="Gabow (greedy ON)")
    xmax = max(off + on)
    for y, v in zip(ys, off):
        ax.text(
            v + xmax * 0.01,
            y + h / 2,
            f"{v:,.3g}x" if v < 10 else f"{v:,.0f}x",
            va="center",
            fontsize=10,
        )
    for y, v in zip(ys, on):
        ax.text(
            v + xmax * 0.01,
            y - h / 2,
            f"{v:,.3g}x" if v < 10 else f"{v:,.0f}x",
            va="center",
            fontsize=10,
        )
    ax.axvline(1, color=ORANGE, linewidth=2, label="1 = same speed as Edmonds")
    ax.set_yticks(ys)
    ax.set_yticklabels([b[0] for b in bars], fontsize=10)
    ax.tick_params(axis="x", labelsize=14)
    ax.grid(axis="x", color="#dddddd")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlim(0, xmax * 1.12)
    ax.set_xlabel("Speedup [times faster than Edmonds]", fontsize=16)
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_int))
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=14, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(OUT / "A4_speedup_summary.png", dpi=200)
    plt.close(fig)
    slower_on = [b[0].split("\n")[0] for b in bars if b[2] < 1]
    slower_off = [b[0].split("\n")[0] for b in bars if b[1] < 1]
    takeaway = (
        "With the greedy start Gabow is faster than Edmonds on every graph type"
        if not slower_on
        else f"With the greedy start Gabow is slower on {', '.join(slower_on)}"
    )
    takeaway += (
        f"; with an empty start it is slower on {_join_types(slower_off)}"
        " (a constant factor of the Python code) and faster everywhere else."
        if slower_off
        else "."
    )
    _caption(
        "A4_speedup_summary",
        title,
        takeaway,
        [
            f"{b[0].replace(chr(10), ' ')}: greedy OFF {b[1]:,.3g}x, greedy ON {b[2]:,.3g}x"
            for b in bars
        ],
    )


# --------------------------------------------------------------------------
# B. Complexity
# --------------------------------------------------------------------------


def iterations_chart(stageb):
    title = "Number of iterations vs graph size"
    sub = "Random sparse graphs, empty start: far below the sqrt(n) growth the bound allows"
    fig, ax = _new_chart(title, sub)
    nums = []
    first = []
    for (c, fam), col in zip(SPARSE, TYPE_COLORS):
        s = _ops_by_n(stageb, fam, lambda r: r["iterations"])
        _line(
            ax,
            list(s),
            list(s.values()),
            {"color": col, "ls": "-", "marker": "o", "label": f"m = {c}n"},
        )
        nums.append(f"m = {c}n: " + ", ".join(f"{n:,}: {v:g}" for n, v in s.items()))
        first.append((min(s), s[min(s)]))
    n0 = first[0][0]
    y0 = max(v for _, v in first)
    xs = [n0 + i * (100000 - n0) / 200 for i in range(201)]
    ax.plot(
        xs,
        [y0 * math.sqrt(x / n0) for x in xs],
        color=GREY,
        ls="--",
        linewidth=2.5,
        label="sqrt(n) growth, from the first point",
    )
    _finish(fig, ax, "B1_iterations_sparse", "Graph size [nodes]", "Iterations")
    nums.append(
        f"sqrt(n) curve: {y0:g} at {n0:,} nodes, {y0 * math.sqrt(100000 / n0):.0f}"
        " at 100,000"
    )
    _caption(
        "B1_iterations_sparse",
        title,
        "On random sparse graphs the number of iterations hardly grows with n,"
        " far below the sqrt(n) curve: random graphs are easy for the algorithm.",
        nums,
    )


def worst_case_chart(f5):
    title = "Worst case: iterations grow like sqrt(n)"
    sub = "Short-and-long chains from the authors' greedy start: points on a straight line"
    fig, ax = _new_chart(title, sub)
    pts = sorted((r["n"], r["iterations"]) for r in f5)
    xs = [math.sqrt(n) for n, _ in pts]
    ys = [it for _, it in pts]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum(
        (x - mx) ** 2 for x in xs
    )
    icpt = my - slope * mx
    fx = [0, max(xs) * 1.05]
    ax.plot(
        fx,
        [icpt + slope * x for x in fx],
        color=GREY,
        ls="--",
        linewidth=2.5,
        label=f"fitted line: {slope:.2f} * sqrt(n) {icpt:+.1f}",
    )
    _line(
        ax,
        xs,
        ys,
        {"color": BLUE, "ls": "-", "marker": "o", "label": "our implementation"},
    )
    paper = R.PAPER_SHORT_AND_LONG
    ax.plot(
        [math.sqrt(n) for n, _, _ in paper],
        [it for _, _, it in paper],
        "*",
        color="black",
        markersize=18,
        label="paper, Table 1 (stopped early by a heuristic)",
    )
    _finish(
        fig,
        ax,
        "B2_worst_case_sqrt",
        "sqrt(graph size)  [sqrt(nodes)]",
        "Iterations",
        int_x=False,
    )
    _caption(
        "B2_worst_case_sqrt",
        title,
        f"On this family the iteration count is a straight line in sqrt(n)"
        f" (about {slope:.2f} * sqrt(n)): the sqrt(n) bound is reached up to a"
        " constant factor. The paper's points lie lower because its driver"
        " switches to a heuristic part-way; they also grow like sqrt(n).",
        [
            "ours: "
            + ", ".join(
                f"{n:,} nodes (sqrt {math.sqrt(n):.0f}): {it}" for n, it in pts
            ),
            "paper Table 1: " + ", ".join(f"{n:,} nodes: {it}" for n, _, it in paper),
            f"fit: iterations = {slope:.3f} * sqrt(n) {icpt:+.2f}",
        ],
    )


def ratio_charts(stageb, name, title, sub, ylabel, f, takeaway):
    for kind, fams in (
        ("sparse", [(f"m = {c}n", fam) for c, fam in SPARSE]),
        ("dense", [(f"density {d}%", fam) for d, fam in DENSE if d in (10, 50, 100)]),
    ):
        t = f"{title} - {'Sparse' if kind == 'sparse' else 'Dense'} graphs"
        fig, ax = _new_chart(t, sub)
        nums = []
        for (label, fam), col in zip(fams, TYPE_COLORS):
            s = _ops_by_n(stageb, fam, f)
            _line(
                ax,
                list(s),
                list(s.values()),
                {"color": col, "ls": "-", "marker": "o", "label": label},
            )
            nums.append(
                f"{label}: " + ", ".join(f"{n:,}: {v:.3g}" for n, v in s.items())
            )
        _finish(fig, ax, f"{name}_{kind}", "Graph size [nodes]", ylabel)
        _caption(f"{name}_{kind}", t, takeaway, nums)


def broom_chart():
    title = "The broom graph, before and after the fix"
    sub = "Steps walked up the search trees: grew like m * L before, zero after"
    fig, ax = _new_chart(title, sub)
    _line(
        ax,
        BROOM["n"],
        [s / 1e6 for s in BROOM["steps_before"]],
        {"color": ORANGE, "ls": "-", "marker": "o", "label": "before the fix"},
    )
    _line(
        ax,
        BROOM["n"],
        [s / 1e6 for s in BROOM["steps_after"]],
        {"color": BLUE, "ls": "-", "marker": "s", "label": "after the fix"},
    )
    _finish(fig, ax, "B5_broom", "Graph size [nodes]", "Steps up the trees [millions]")
    b, a = BROOM_TIME_L160
    _caption(
        "B5_broom",
        title,
        "Before the fix every edge between two search trees walked up both trees,"
        " which grows faster than the bound; the fix decides this in one step, so"
        f" the walk disappears (at {BROOM['n'][-1]:,} nodes the run took {b} s"
        f" before and {a} s after).",
        [
            f"{n:,} nodes (m = {m:,}): before {sb:,} steps, after {sa}"
            for n, m, sb, sa in zip(
                BROOM["n"], BROOM["m"], BROOM["steps_before"], BROOM["steps_after"]
            )
        ],
    )


# --------------------------------------------------------------------------


def write_captions():
    lines = [
        "# Presentation charts: captions\n",
        (
            "Generated by `benchmarks/presentation_plots.py` from the CSVs in"
            " `benchmarks/results/`. Times: median over graphs of each graph's median"
            " run; speedups: median of per-graph speedups; operation counts: median"
            " over seeds, empty start (greedy OFF).\n"
        ),
    ]
    for name, title, takeaway, numbers in captions:
        lines.append(f"## {title}\n")
        lines.append(f"File: `{name}.png`\n")
        lines.append(f"**Takeaway:** {takeaway}\n")
        lines.append("Numbers shown:\n")
        lines.extend(f"- {x}" for x in numbers)
        lines.append("")
    (OUT / "captions.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stagec = R.load_stagec_csv()
    hard = R._apply_f5_idle_rerun(R._load_hard_rows())
    stageb = R.load_stageb_csv()
    f5 = R.load_f5_edgescan_csv()

    comparison_charts(stagec, hard)
    speedup_bars(stagec, hard)
    iterations_chart(stageb)
    worst_case_chart(f5)
    ratio_charts(
        stageb,
        "B3_total_work",
        "Total work relative to the bound",
        "Total operations / (sqrt(n) * m): flat or falling means within the bound",
        "Operations / (sqrt(n) * m)",
        lambda r: r["total_ops"] / (math.sqrt(r["n"]) * r["m"]),
        "Total work divided by sqrt(n) * m stays flat or falls as the graphs grow,"
        " so the measured work stays within the O(sqrt(n) * m) bound.",
    )
    ratio_charts(
        stageb,
        "B4_work_per_iteration",
        "Work per iteration",
        "Operations per iteration / m: flat means each iteration costs O(m)",
        "Operations per iteration / m",
        lambda r: r["total_ops"] / r["iterations"] / r["m"],
        "Work per iteration divided by m stays in a narrow band, so one"
        " iteration costs time proportional to the number of edges.",
    )
    broom_chart()
    write_captions()
    for name, *_ in captions:
        print(OUT / f"{name}.png")
    print(OUT / "captions.md")


if __name__ == "__main__":
    main()
