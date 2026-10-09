"""Slide-ready charts for the presentation, built from the existing results.

Reads only the CSVs that ``run_all.py`` already wrote to ``results/`` (no new
runs) and writes one PNG per chart plus ``captions.md`` to
``results/presentation/``. Re-run it after ``run_all.py`` to refresh the
charts::

    python benchmarks/presentation_plots.py

The one exception is chart B5b (broom graph, running time before and after
the Phase 1 climb fix), which needs ``results/broom_timing.csv``; create it
once with ``--measure-broom`` (times the current code against
``matching.py`` at tag ``before-climb-fix``, about a minute).

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

import networkx as nx

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


def _short(k):
    return {
        "gabow_on": "Gabow",
        "gabow_off": "Gabow (greedy OFF)",
        "gabow_cpp": "Gabow (authors' start)",
        "edmonds": "Edmonds",
    }[k]


def _secs_label(v):
    return f"{v:.2g} s" if v < 10 else f"{v:,.0f} s"


def _all_greedy_optimal(stagec, family, n):
    """True if the greedy start alone was already maximum on every graph of
    this point (greedy ON runs)."""
    flags = {
        r["seed"]: r["greedy_already_optimal"]
        for r in stagec
        if r["family"] == family
        and r["n"] == n
        and r["algorithm"] == "gabow"
        and r["greedy_init"] is True
    }
    return bool(flags) and all(flags.values())


def time_chart(
    rows, family, title, name, variants, max_n=None, notes=(), label_last=False
):
    """One running-time chart. `max_n` drops larger sizes; `notes` is a list
    of (variant, n, text) annotations; `label_last` writes the time next to
    the last point of each line."""
    fig, ax = _new_chart(title)
    series = {}
    for k in variants:
        s = _time_series(rows, family, k)
        series[k] = {n: v for n, v in s.items() if max_n is None or n <= max_n}
    for k in variants:
        s = series[k]
        if s:
            _line(ax, list(s), list(s.values()), k)
    for k, n, text in notes:
        ax.annotate(
            text,
            xy=(n, series[k][n]),
            xytext=(-20, 45),
            textcoords="offset points",
            ha="right",
            fontsize=14,
            color=STYLE[k]["color"],
            arrowprops={"arrowstyle": "->", "color": STYLE[k]["color"], "lw": 1.5},
        )
    if label_last:
        offsets = {"gabow_on": (-12, 8), "gabow_off": (-12, 30), "edmonds": (-12, -6)}
        for k in variants:
            n = max(series[k])
            dx, dy = offsets.get(k, (-12, 8))
            ax.annotate(
                f"{_short(k)}: {_secs_label(series[k][n])}",
                xy=(n, series[k][n]),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="right",
                va="top" if dy < 0 else "bottom",
                fontsize=14,
                fontweight="bold",
                color=STYLE[k]["color"],
            )
    _finish(fig, ax, name, "Graph size [nodes]", "Running time [sec]")
    nums = []
    for k in variants:
        pts = ", ".join(f"{n:,}: {_sec(v)}" for n, v in series[k].items())
        nums.append(f"{STYLE[k]['label']}: {pts}")
    nums.extend(f"Note at {n:,} nodes ({_short(k)}): {text}" for k, n, text in notes)
    big = max(series["edmonds"])
    parts = [
        f"{_short(k)} {_sec(series[k].get(big))}"
        for k in variants
        if series[k].get(big) is not None
    ]
    _caption(name, title, f"At {big:,} nodes: " + "; ".join(parts) + ".", nums)
    return series


def _add_takeaway(text):
    captions[-1] = captions[-1][:2] + (captions[-1][2] + text, captions[-1][3])


def _add_numbers(lines):
    captions[-1][3].extend(lines)


def comparison_charts(stagec, hard):
    for d, fam in DENSE:
        n_big = max(r["n"] for r in stagec if r["family"] == fam)
        notes = []
        if d == 75 and _all_greedy_optimal(stagec, fam, n_big):
            notes.append(("gabow_on", n_big, "greedy alone already optimal"))
        time_chart(
            stagec,
            fam,
            f"Running Time - Density {d}%",
            f"A1_dense_{d}",
            ["gabow_on", "gabow_off", "edmonds"],
            notes=notes,
        )
        on = _speedup(stagec, fam, "gabow_on", n_big)
        off = _speedup(stagec, fam, "gabow_off", n_big)
        _add_takeaway(
            f" Gabow is {on:.2g}x as fast as Edmonds with the greedy start and"
            f" {off:.2g}x without it (median of per-graph speedups)."
            + (
                " At this size the greedy start alone was already a maximum"
                " matching on every graph."
                if notes
                else ""
            )
        )
    for c, fam in SPARSE:
        n_ed = max(
            r["n"] for r in stagec if r["family"] == fam and r["algorithm"] == "edmonds"
        )
        time_chart(
            stagec,
            fam,
            f"Running Time - Sparse graphs (m = {c}n)",
            f"A2_sparse_{c}n",
            ["gabow_on", "gabow_off", "edmonds"],
            max_n=n_ed,
        )
        on = _speedup(stagec, fam, "gabow_on", n_ed)
        off = _speedup(stagec, fam, "gabow_off", n_ed)
        full_on = _time_series(stagec, fam, "gabow_on")
        full_off = _time_series(stagec, fam, "gabow_off")
        n_max = max(full_on)
        _add_takeaway(
            f" Gabow is {on:,.0f}x faster with the greedy start and {off:,.0f}x"
            f" without it. Edmonds was not run beyond {n_ed:,} nodes (too slow);"
            f" Gabow was also run up to {n_max:,} nodes: {_sec(full_on[n_max])}"
            f" with the greedy start, {_sec(full_off[n_max])} without it."
        )
        _add_numbers(
            [
                f"Not on the chart (no Edmonds run): Gabow (greedy ON) "
                + ", ".join(f"{n:,}: {_sec(v)}" for n, v in full_on.items() if n > n_ed)
                + "; Gabow (greedy OFF) "
                + ", ".join(
                    f"{n:,}: {_sec(v)}" for n, v in full_off.items() if n > n_ed
                )
            ]
        )
    for fam, title, name in HARD:
        variants = ["gabow_on", "gabow_off", "edmonds"]
        if fam == "F5_chains_edgescan_hard":
            variants = ["gabow_on", "gabow_cpp", "edmonds"]
        s = time_chart(
            hard,
            fam,
            f"Running Time - {title}",
            f"A3_{name}",
            variants,
            label_last=fam == "F6_long_path",
        )
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
        _add_takeaway(extra)


def _fmt_x(v):
    return f"{v:.2g}x" if v < 10 else f"{v:,.0f}x"


def bar_chart(name, title, subtitle, bars, footnote=None):
    """Horizontal grouped bars. bars: (label, off, on, note_on)."""
    fig, ax = plt.subplots(figsize=(12, 6.75))
    fig.suptitle(title, fontsize=20, fontweight="bold", y=0.97)
    ax.set_title(subtitle, fontsize=14, color="#444444", pad=12)
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
    pad = xmax * 0.01
    for y, b in zip(ys, bars):
        ax.text(b[1] + pad, y + h / 2, _fmt_x(b[1]), va="center", fontsize=13)
        text = _fmt_x(b[2]) + (f"  ({b[3]})" if b[3] else "")
        ax.text(b[2] + pad, y - h / 2, text, va="center", fontsize=13)
    ax.axvline(1, color=ORANGE, linewidth=3, label="1 = same speed as Edmonds")
    ax.text(
        1,
        ys[0] + 0.75,
        " 1x",
        color=ORANGE,
        fontsize=14,
        fontweight="bold",
        va="bottom",
    )
    ax.set_yticks(ys)
    ax.set_yticklabels([b[0] for b in bars], fontsize=14)
    ax.tick_params(axis="x", labelsize=14)
    ax.grid(axis="x", color="#dddddd")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlim(0, xmax * 1.3)
    ax.set_ylim(-0.7, len(bars) - 0.1)
    ax.set_xlabel("Speedup [times faster than Edmonds]", fontsize=16)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:,.0f}"))
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=14, frameon=False)
    bottom = 0.08
    if footnote:
        fig.text(0.01, 0.075, footnote, fontsize=12, color="#444444", ha="left")
        bottom = 0.12
    fig.tight_layout(rect=(0, bottom, 1, 0.95))
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)


def _stable_speedups(stable_rows, c):
    """Median over seeds of (Edmonds median / Gabow median) on the same graph,
    for greedy OFF and ON, from the --stable-check runs."""
    out = {}
    for on in (False, True):
        per_seed = []
        for seed in sorted({r["seed"] for r in stable_rows if r["c"] == c}):
            e = [
                r["runtime_sec"]
                for r in stable_rows
                if r["c"] == c and r["seed"] == seed and r["algorithm"] == "edmonds"
            ]
            g = [
                r["runtime_sec"]
                for r in stable_rows
                if r["c"] == c
                and r["seed"] == seed
                and r["algorithm"] == "gabow"
                and r["greedy_init"] == str(on)
            ]
            per_seed.append(statistics.median(e) / statistics.median(g))
        out[on] = (statistics.median(per_seed), min(per_seed), max(per_seed))
    return out


def _load_stable():
    import csv

    with open(R.OUTDIR / "stageC_stable_check.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["c"] = int(r["c"])
        r["n"] = int(r["n"])
        r["seed"] = int(r["seed"])
        r["runtime_sec"] = float(r["runtime_sec"])
    return rows


def speedup_bars(stagec, hard):
    # A4a: sparse (stable check) and hard families.
    stable = _load_stable()
    n_stable = stable[0]["n"]
    bars, nums = [], []
    for c, _fam in SPARSE:
        sp = _stable_speedups(stable, c)
        n_seeds = len({r["seed"] for r in stable if r["c"] == c})
        bars.append(
            (
                f"Random sparse, m = {c}n\n({n_stable:,} nodes)",
                sp[False][0],
                sp[True][0],
                "",
            )
        )
        nums.append(
            f"Random sparse m = {c}n, {n_stable:,} nodes (stable check, median of"
            f" {n_seeds} graphs): greedy OFF {_fmt_x(sp[False][0])}"
            f" [{_fmt_x(sp[False][1])}-{_fmt_x(sp[False][2])}], greedy ON"
            f" {_fmt_x(sp[True][0])} [{_fmt_x(sp[True][1])}-{_fmt_x(sp[True][2])}]"
        )
    starred = False
    for fam, title, _ in HARD:
        ed = [r for r in hard if r["family"] == fam and r["algorithm"] == "edmonds"]
        n = max(r["n"] for r in ed)
        single = sum(1 for r in ed if r["n"] == n) == 1
        starred |= single
        label = title.split(" (")[0] + ("*" if single else "")
        off = _speedup(hard, fam, "gabow_off", n)
        on = _speedup(hard, fam, "gabow_on", n)
        bars.append((f"{label}\n({n:,} nodes)", off, on, ""))
        nums.append(
            f"{label}, {n:,} nodes: greedy OFF {_fmt_x(off)}, greedy ON {_fmt_x(on)}"
            + (" (single Edmonds run)" if single else "")
        )
    foot = (
        "* based on a single Edmonds run (Edmonds is too slow at this size to"
        " repeat); Gabow times are medians of repeated runs."
        if starred
        else None
    )
    title = "How many times faster than Edmonds - Sparse and hard graphs"
    bar_chart(
        "A4a_speedup_sparse_hard",
        title,
        "Largest size where both ran (sparse: repeated runs); above 1 = Gabow is faster",
        bars,
        foot,
    )
    lo = min(min(b[1], b[2]) for b in bars)
    _caption(
        "A4a_speedup_sparse_hard",
        title,
        f"On sparse and hard graphs Gabow is faster than Edmonds everywhere"
        f" (at least {_fmt_x(lo)}), with or without the greedy start.",
        nums + ([foot] if foot else []),
    )

    # A4b: dense random graphs.
    bars, nums = [], []
    for d, fam in DENSE:
        n = max(
            r["n"] for r in stagec if r["family"] == fam and r["algorithm"] == "edmonds"
        )
        off = _speedup(stagec, fam, "gabow_off", n)
        on = _speedup(stagec, fam, "gabow_on", n)
        note = (
            "greedy alone already optimal"
            if _all_greedy_optimal(stagec, fam, n)
            else ""
        )
        bars.append((f"Density {d}%\n({n:,} nodes)", off, on, note))
        nums.append(
            f"Density {d}%, {n:,} nodes: greedy OFF {_fmt_x(off)}, greedy ON"
            f" {_fmt_x(on)}" + (f" ({note} on all graphs)" if note else "")
        )
    title = "How many times faster than Edmonds - Dense random graphs"
    bar_chart(
        "A4b_speedup_dense",
        title,
        "At 800 nodes; above 1 = Gabow is faster, below 1 = Edmonds is faster",
        bars,
    )
    slow = [1 / b[1] for b in bars if b[1] < 1]
    _caption(
        "A4b_speedup_dense",
        title,
        "On dense graphs Gabow wins only with the greedy start; from an empty"
        f" start Edmonds is {min(slow):.2g}-{max(slow):.2g} times faster (a"
        " constant factor of the Python code). The largest wins are where the"
        " greedy start alone was already a maximum matching.",
        nums,
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


BROOM_TIMING_CSV = "broom_timing.csv"
BROOM_LS = (20, 40, 80, 120, 160)


def _broom(L, k):
    """Same graph as TestGabow._broom in test_matching.py: from each of two
    free roots a matched path of length 2L ends in an even vertex with k
    pendant matched edges; K_{k,k} joins the leaves. Returns (G, matching)."""
    G = nx.Graph()
    M = {}

    def tree(tag):
        prev = (tag, "r")
        for j in range(L):
            o, e = (tag, "o", j), (tag, "e", j)
            G.add_edge(prev, o)
            G.add_edge(o, e)
            M[o], M[e] = e, o
            prev = e
        leaves = []
        for j in range(k):
            a, b = (tag, "a", j), (tag, "b", j)
            G.add_edge(prev, a)
            G.add_edge(a, b)
            M[a], M[b] = b, a
            leaves.append(b)
        return leaves

    A, B = tree("A"), tree("B")
    G.add_edges_from((x, y) for x in A for y in B)
    return G, M


def measure_broom(reps=5, tag="before-climb-fix"):
    """Time the broom family with the current code and with matching.py at
    `tag` (the state before the climb fix), interleaved, counters off, from
    the broom's own starting matching. Writes results/broom_timing.csv."""
    import csv
    import gc
    import importlib.util
    import subprocess
    import tempfile
    import time

    import networkx.utils.backends as nxb

    repo = Path(__file__).resolve().parent.parent
    src = subprocess.run(
        ["git", "show", f"{tag}:networkx/algorithms/matching.py"],
        cwd=repo,
        capture_output=True,
        check=True,
    ).stdout
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "_matching_before_fix.py"
        path.write_bytes(src)
        spec = importlib.util.spec_from_file_location("_matching_before_fix", path)
        old = importlib.util.module_from_spec(spec)
        # The old module registers the same function names with NetworkX's
        # dispatcher; load it with an empty registry, then restore it.
        registry = nxb._registered_algorithms
        saved = dict(registry)
        registry.clear()
        try:
            spec.loader.exec_module(old)
        finally:
            registry.clear()
            registry.update(saved)
    fns = {
        "before": old.max_cardinality_matching_gabow,
        "after": nx.max_cardinality_matching_gabow,
    }
    rows = []
    for L in BROOM_LS:
        G, M = _broom(L, L)
        sizes = set()
        for rep in range(reps):
            for version, f in fns.items():
                gc.collect()
                t0 = time.perf_counter()
                res = f(G, _skip_greedy_init=True, _initial_mate=dict(M))
                dt = time.perf_counter() - t0
                sizes.add(len(res))
                rows.append(
                    {
                        "L": L,
                        "n": G.number_of_nodes(),
                        "m": G.number_of_edges(),
                        "version": version,
                        "rep": rep,
                        "runtime_sec": dt,
                        "size": len(res),
                    }
                )
        assert len(sizes) == 1, f"before/after sizes differ at L={L}: {sizes}"
        print(f"broom L={L}: done", flush=True)
    with open(R.OUTDIR / BROOM_TIMING_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def broom_time_chart():
    import csv

    path = R.OUTDIR / BROOM_TIMING_CSV
    if not path.exists():
        print(f"skipping B5b: {path.name} missing (run with --measure-broom)")
        return
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    ns = sorted({int(r["n"]) for r in rows})
    med = {
        (v, n): statistics.median(
            float(r["runtime_sec"])
            for r in rows
            if r["version"] == v and int(r["n"]) == n
        )
        for v in ("before", "after")
        for n in ns
    }
    title = "The broom graph, before and after the fix"
    sub = "Running time: the walk up the trees dominated before the fix"
    fig, ax = _new_chart(title, sub)
    _line(
        ax,
        ns,
        [med["before", n] for n in ns],
        {"color": ORANGE, "ls": "-", "marker": "o", "label": "before the fix"},
    )
    _line(
        ax,
        ns,
        [med["after", n] for n in ns],
        {"color": BLUE, "ls": "-", "marker": "s", "label": "after the fix"},
    )
    _finish(fig, ax, "B5b_broom_time", "Graph size [nodes]", "Running time [sec]")
    reps = len(rows) // (2 * len(ns))
    big = ns[-1]
    _caption(
        "B5b_broom_time",
        title + " (running time)",
        f"At {big:,} nodes the fix cuts the running time from"
        f" {_sec(med['before', big])} to {_sec(med['after', big])}"
        f" ({med['before', big] / med['after', big]:.0f}x); both versions return"
        " the same matching size.",
        [
            f"{n:,} nodes: before {_sec(med['before', n])}, after"
            f" {_sec(med['after', n])}"
            for n in ns
        ]
        + [f"median of {reps} runs each, counters off, broom's own starting matching"],
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


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--measure-broom",
        action="store_true",
        help="first time the broom family before/after the climb fix (about a"
        " minute) and write results/broom_timing.csv for chart B5b",
    )
    args = p.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.measure_broom:
        measure_broom()
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
    broom_time_chart()
    write_captions()
    for name, *_ in captions:
        print(OUT / f"{name}.png")
    print(OUT / "captions.md")


if __name__ == "__main__":
    main()
