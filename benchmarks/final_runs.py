"""Final experiments for the presentation: Gabow vs Edmonds on random sparse
and dense graphs, plus operation counts, with charts generated at the end.

Resumable: every result is appended to a CSV in ``results/final/`` as soon as
it is measured; on restart, finished rows are skipped. Run::

    python benchmarks/final_runs.py              # full run, then charts
    python benchmarks/final_runs.py --tiny       # small smoke test (results/final_tiny/)
    python benchmarks/final_runs.py --charts-only

Graphs: 10 per (family, size), seeds 0-9; every algorithm runs on the same
graph and the matching sizes are compared.

- Sparse ``gnm_random_graph(n, c*n)``, c = 3, 5, 10, n = 500 ... 20,000;
  Gabow only at 50,000 and 100,000.
- Dense ``gnp_random_graph(n, p)``, p = 0.25, 0.5, 0.75, 1.0,
  n = 100 ... 2,000 (p = 1.0 is the complete graph, the same for every seed).

Per graph:

- Time, counters off: Edmonds (``max_weight_matching(G, maxcardinality=True)``,
  in a worker process so a call can be stopped after EDMONDS_TIMEOUT), then
  Gabow greedy ON (default) and greedy OFF, back to back. A call that takes
  under REPEAT_BELOW seconds is repeated REPEATS times and the median kept.
- Operation counts, counters on, once each for greedy ON and OFF (they are
  deterministic): iterations and total ops (sum of verify_complexity.OP_KEYS).

Order: size index by size index across all families (smallest first); the
dense 2,000 point, then the Gabow-only sparse sizes, and the sparse Edmonds
runs at 15,000 and 20,000 last. If an Edmonds call times out, Edmonds is
skipped for the rest of that size and all larger sizes of that family.
"""

import argparse
import csv
import math
import multiprocessing as mp
import os
import statistics
import sys
import time
from pathlib import Path

import networkx as nx

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

SEEDS = range(10)
SPARSE_CS = (3, 5, 10)
SPARSE_SIZES = (500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000)
SPARSE_GABOW_ONLY = (50000, 100000)
DENSE_PS = (0.25, 0.5, 0.75, 1.0)
DENSE_SIZES = (100, 200, 400, 600, 800, 1000, 1500, 2000)
EDMONDS_TIMEOUT = 15 * 60
REPEAT_BELOW = 1.0
REPEATS = 5

TIMES_FIELDS = [
    "family",
    "n",
    "m",
    "seed",
    "algorithm",
    "status",
    "runtime_sec",
    "reps",
    "size",
]
OPS_FIELDS = [
    "family",
    "n",
    "m",
    "seed",
    "greedy",
    "iterations",
    "total_ops",
    "augmentations",
    "size",
]

OP_KEYS = [
    "edge_scans",
    "search_steps",
    "blossom_shrink_vertex_steps",
    "uf_base_find_hops",
    "uf_dbase_find_hops",
    "uf_base_union_calls",
    "uf_dbase_union_calls",
    "augmentations",
    "climb_steps_blossom",
    "climb_steps_cross_tree",
]


# --------------------------------------------------------------------------
# Graphs and timing
# --------------------------------------------------------------------------


def family_name(kind, param):
    if kind == "sparse":
        return f"sparse_m{param}n"
    return f"dense_p{round(param * 100)}"


def parse_family(family):
    if family.startswith("sparse_m"):
        return "sparse", int(family[len("sparse_m") : -1])
    return "dense", int(family[len("dense_p") :]) / 100


def build_graph(family, n, seed):
    kind, param = parse_family(family)
    if kind == "sparse":
        return nx.gnm_random_graph(n, param * n, seed=seed)
    return nx.gnp_random_graph(n, param, seed=seed)


def timed(f):
    """Median time of f() (repeated if fast) and its result."""
    t0 = time.perf_counter()
    res = f()
    times = [time.perf_counter() - t0]
    if times[0] < REPEAT_BELOW:
        for _ in range(REPEATS - 1):
            t0 = time.perf_counter()
            f()
            times.append(time.perf_counter() - t0)
    return statistics.median(times), len(times), res


def _edmonds_worker(conn):
    """Child process: build the graph, report "built", time Edmonds."""
    while True:
        job = conn.recv()
        if job is None:
            return
        family, n, seed = job
        G = build_graph(family, n, seed)
        conn.send(("built",))
        t, reps, M = timed(lambda G=G: nx.max_weight_matching(G, maxcardinality=True))
        conn.send(("done", t, reps, len(M)))
        del G, M


class EdmondsRunner:
    """Edmonds in a persistent worker process, killed and restarted on a
    timeout."""

    def __init__(self):
        self.proc = None
        self.conn = None

    def _start(self):
        parent, child = mp.Pipe()
        self.proc = mp.Process(target=_edmonds_worker, args=(child,), daemon=True)
        self.proc.start()
        self.conn = parent

    def run(self, family, n, seed, timeout):
        if self.proc is None or not self.proc.is_alive():
            self._start()
        self.conn.send((family, n, seed))
        self.conn.recv()  # "built": the timeout covers Edmonds only
        if self.conn.poll(timeout):
            _tag, t, reps, size = self.conn.recv()
            return "ok", t, reps, size
        self.proc.kill()
        self.proc.join()
        self.proc = None
        return "timeout", None, 0, None

    def close(self):
        if self.proc is not None and self.proc.is_alive():
            self.conn.send(None)
            self.proc.join(5)


# --------------------------------------------------------------------------
# Resumable CSVs
# --------------------------------------------------------------------------


class Store:
    def __init__(self, outdir):
        self.outdir = outdir
        outdir.mkdir(parents=True, exist_ok=True)
        self.times_path = outdir / "times.csv"
        self.ops_path = outdir / "ops.csv"
        self.times = self._load(self.times_path, TIMES_FIELDS)
        self.ops = self._load(self.ops_path, OPS_FIELDS)
        self.time_keys = {
            (r["family"], int(r["n"]), int(r["seed"]), r["algorithm"])
            for r in self.times
        }
        self.ops_keys = {
            (r["family"], int(r["n"]), int(r["seed"]), r["greedy"]) for r in self.ops
        }
        # Edmonds disabled from this n upward, per family (after a timeout).
        self.edmonds_off = {}
        for r in self.times:
            if r["algorithm"] == "edmonds" and r["status"] == "timeout":
                n = int(r["n"])
                f = r["family"]
                self.edmonds_off[f] = min(n, self.edmonds_off.get(f, n))

    @staticmethod
    def _load(path, fields):
        if not path.exists():
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writeheader()
            return []
        with open(path, newline="") as f:
            return list(csv.DictReader(f))

    @staticmethod
    def _append(path, fields, row):
        with open(path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
            f.flush()
            os.fsync(f.fileno())

    def add_time(self, row):
        self._append(self.times_path, TIMES_FIELDS, row)
        self.times.append({k: str(v) for k, v in row.items()})
        self.time_keys.add((row["family"], row["n"], row["seed"], row["algorithm"]))

    def add_ops(self, row):
        self._append(self.ops_path, OPS_FIELDS, row)
        self.ops.append({k: str(v) for k, v in row.items()})
        self.ops_keys.add((row["family"], row["n"], row["seed"], row["greedy"]))


# --------------------------------------------------------------------------
# Plan and run
# --------------------------------------------------------------------------


def plan(cfg):
    """[(family, n, with_edmonds)] in run order."""
    sparse = [family_name("sparse", c) for c in cfg["sparse_cs"]]
    dense = [family_name("dense", p) for p in cfg["dense_ps"]]
    s_sizes, d_sizes = cfg["sparse_sizes"], cfg["dense_sizes"]
    late = set(s_sizes[-cfg["sparse_late"] :]) if cfg["sparse_late"] else set()
    s_early = [n for n in s_sizes if n not in late]
    steps = []
    for i in range(max(len(s_early), len(d_sizes))):
        if i < len(s_early):
            steps += [(f, s_early[i], True) for f in sparse]
        if i < len(d_sizes):
            steps += [(f, d_sizes[i], True) for f in dense]
    for n in cfg["sparse_gabow_only"]:
        steps += [(f, n, False) for f in sparse]
    for n in sorted(late):
        steps += [(f, n, True) for f in sparse]
    return steps


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def run(cfg, store):
    runner = EdmondsRunner()
    mismatches = 0
    try:
        for family, n, with_edmonds in plan(cfg):
            for seed in cfg["seeds"]:
                need_t = [
                    a
                    for a in ("edmonds", "gabow_on", "gabow_off")
                    if (family, n, seed, a) not in store.time_keys
                    and (a != "edmonds" or with_edmonds)
                ]
                need_o = [
                    g
                    for g in ("on", "off")
                    if (family, n, seed, g) not in store.ops_keys
                ]
                if not need_t and not need_o:
                    continue
                G = build_graph(family, n, seed)
                m = G.number_of_edges()
                base = {"family": family, "n": n, "m": m, "seed": seed}
                sizes = {}
                if "edmonds" in need_t:
                    off_from = store.edmonds_off.get(family)
                    if off_from is not None and n >= off_from:
                        pass  # disabled after a timeout at a size <= n
                    else:
                        status, t, reps, size = runner.run(
                            family, n, seed, cfg["edmonds_timeout"]
                        )
                        store.add_time(
                            {
                                **base,
                                "algorithm": "edmonds",
                                "status": status,
                                "runtime_sec": "" if t is None else f"{t:.6f}",
                                "reps": reps,
                                "size": "" if size is None else size,
                            }
                        )
                        if status == "timeout":
                            store.edmonds_off[family] = n
                            log(f"Edmonds TIMEOUT {family} n={n} seed={seed}")
                        else:
                            sizes["edmonds"] = size
                for alg, kw in (
                    ("gabow_on", {}),
                    ("gabow_off", {"_skip_greedy_init": True}),
                ):
                    if alg not in need_t:
                        continue
                    t, reps, M = timed(
                        lambda kw=kw, G=G: nx.max_cardinality_matching_gabow(G, **kw)
                    )
                    sizes[alg] = len(M)
                    store.add_time(
                        {
                            **base,
                            "algorithm": alg,
                            "status": "ok",
                            "runtime_sec": f"{t:.6f}",
                            "reps": reps,
                            "size": len(M),
                        }
                    )
                for g in need_o:
                    c = {}
                    M = nx.max_cardinality_matching_gabow(
                        G, _counters=c, _skip_greedy_init=(g == "off")
                    )
                    sizes[f"ops_{g}"] = len(M)
                    store.add_ops(
                        {
                            **base,
                            "greedy": g,
                            "iterations": c["iterations"],
                            "total_ops": sum(c[k] for k in OP_KEYS),
                            "augmentations": c["augmentations"],
                            "size": len(M),
                        }
                    )
                if len(set(sizes.values())) > 1:
                    mismatches += 1
                    log(f"SIZE MISMATCH {family} n={n} seed={seed}: {sizes}")
                del G
            log(f"done {family} n={n}")
    finally:
        runner.close()
    log(f"collection finished; size mismatches: {mismatches}")


# --------------------------------------------------------------------------
# Charts and summary
# --------------------------------------------------------------------------


def _num(v):
    return float(v) if v not in ("", None) else None


def load(outdir):
    with open(outdir / "times.csv", newline="") as f:
        times = list(csv.DictReader(f))
    with open(outdir / "ops.csv", newline="") as f:
        ops = list(csv.DictReader(f))
    for r in times + ops:
        r["n"], r["m"], r["seed"] = int(r["n"]), int(r["m"]), int(r["seed"])
    for r in times:
        r["runtime_sec"] = _num(r["runtime_sec"])
    for r in ops:
        r["iterations"] = int(r["iterations"])
        r["total_ops"] = int(r["total_ops"])
    return times, ops


def _stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return statistics.median(vals), min(vals), max(vals), len(vals)


def per_graph(times, family, n, alg):
    return {
        r["seed"]: r["runtime_sec"]
        for r in times
        if r["family"] == family
        and r["n"] == n
        and r["algorithm"] == alg
        and r["status"] == "ok"
    }


def ops_per_graph(ops, family, n, greedy):
    return {
        r["seed"]: r
        for r in ops
        if r["family"] == family and r["n"] == n and r["greedy"] == greedy
    }


def family_title(family):
    kind, param = parse_family(family)
    if kind == "sparse":
        return f"Sparse graphs (m = {param}n)"
    return f"Dense graphs (density {round(param * 100)}%)"


def make_charts(outdir):
    sys.path.insert(0, str(HERE))
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import presentation_plots as pp
    from matplotlib.ticker import FuncFormatter

    charts = outdir / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    times, ops = load(outdir)
    families = sorted(
        {r["family"] for r in times}, key=lambda f: (f.startswith("dense"), f)
    )
    families.sort(key=lambda f: (parse_family(f)[0] == "dense", parse_family(f)[1]))
    sizes = {f: sorted({r["n"] for r in times if r["family"] == f}) for f in families}
    m_of = {}
    for r in times:
        m_of.setdefault((r["family"], r["n"]), []).append(r["m"])

    alg_style = {
        "gabow_on": pp.STYLE["gabow_on"],
        "gabow_off": pp.STYLE["gabow_off"],
        "edmonds": pp.STYLE["edmonds"],
    }
    type_colors = [*pp.TYPE_COLORS, "#c1121f"]

    def errline(ax, xs, st, style, label):
        med = [s[0] for s in st]
        lo = [s[0] - s[1] for s in st]
        hi = [s[2] - s[0] for s in st]
        ax.errorbar(
            xs,
            med,
            yerr=[lo, hi],
            color=style["color"],
            linestyle=style["ls"],
            marker=style["marker"],
            linewidth=3,
            markersize=8,
            capsize=4,
            elinewidth=1.5,
            label=label,
        )

    def finish(fig, ax, name, xlabel, ylabel, log=False):
        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        if log:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(axis="y", which="both", color="#eeeeee")
        else:
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:,.0f}"))
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=1 if max(map(len, labels)) > 30 else min(len(labels), 3),
            fontsize=13,
            frameon=False,
        )
        rows = len(labels) if max(map(len, labels)) > 30 else math.ceil(len(labels) / 3)
        fig.tight_layout(rect=(0, 0.06 + 0.04 * rows, 1, 0.95))
        fig.savefig(charts / f"{name}.png", dpi=200)
        plt.close(fig)

    def slope(xs, ys):
        pairs = [(x, y) for x, y in zip(xs, ys) if y and y > 0]
        if len(pairs) < 2:
            return float("nan")
        lx, ly = np.log([p[0] for p in pairs]), np.log([p[1] for p in pairs])
        return float(np.polyfit(lx, ly, 1)[0])

    # 1. Time vs n, linear and log-log.
    for fam in families:
        for log_axes in (False, True):
            fig, ax = pp._new_chart(f"Running Time - {family_title(fam)}")
            for alg in ("gabow_on", "gabow_off", "edmonds"):
                xs, st = [], []
                for n in sizes[fam]:
                    s = _stats(per_graph(times, fam, n, alg).values())
                    if s:
                        xs.append(n)
                        st.append(s)
                if not xs:
                    continue
                label = alg_style[alg]["label"]
                if log_axes:
                    k = slope(xs, [s[0] for s in st])
                    if not math.isnan(k):
                        label += f"  (fitted slope {k:.2f})"
                errline(ax, xs, st, alg_style[alg], label)
            suffix = "loglog" if log_axes else "linear"
            finish(
                fig,
                ax,
                f"time_{fam}_{suffix}",
                "Graph size [nodes]",
                "Running time [sec]",
                log=log_axes,
            )

    # 2. Bound charts (ops and time), sparse and dense, greedy ON and OFF.
    def bound_series(fam, greedy, f):
        xs, st = [], []
        for n in sizes[fam]:
            og = ops_per_graph(ops, fam, n, greedy)
            tg = per_graph(times, fam, n, "gabow_" + greedy)
            vals = [f(og[s], tg.get(s)) for s in og]
            s = _stats(vals)
            if s:
                xs.append(n)
                st.append(s)
        return xs, st

    kinds = {
        "sparse": [f for f in families if f.startswith("sparse")],
        "dense": [f for f in families if f.startswith("dense")],
    }
    bound_specs = [
        (
            "a_iterations",
            "Number of iterations vs graph size",
            "dashed grey: sqrt(n) growth from the first point",
            "Iterations",
            lambda o, t: o["iterations"],
        ),
        (
            "b_work_per_iteration_ops",
            "Work per iteration / m (operation counts)",
            "flat = each iteration costs O(m)",
            "Operations per iteration / m",
            lambda o, t: o["total_ops"] / o["iterations"] / o["m"],
        ),
        (
            "c_work_bound_ops",
            "Total work / (sqrt(n) * m) (operation counts)",
            "flat or falling = within the O(sqrt(n) * m) bound",
            "Operations / (sqrt(n) * m)",
            lambda o, t: o["total_ops"] / (math.sqrt(o["n"]) * o["m"]),
        ),
        (
            "b_work_per_iteration_time",
            "Time per iteration / m",
            "flat = each iteration costs O(m) time",
            "Time per iteration / m [microsec]",
            lambda o, t: None if t is None else 1e6 * t / o["iterations"] / o["m"],
        ),
        (
            "c_work_bound_time",
            "Time / (sqrt(n) * m)",
            "flat or falling = running time within the O(sqrt(n) * m) bound",
            "Time / (sqrt(n) * m) [microsec]",
            lambda o, t: None if t is None else 1e6 * t / (math.sqrt(o["n"]) * o["m"]),
        ),
    ]
    for kind, fams in kinds.items():
        if not fams:
            continue
        for greedy in ("on", "off"):
            gtxt = "greedy ON" if greedy == "on" else "greedy OFF (empty start)"
            for key, title, sub, ylabel, f in bound_specs:
                fig, ax = pp._new_chart(
                    f"{title} - {kind.capitalize()} graphs", f"Gabow {gtxt}; {sub}"
                )
                first = None
                for fam, col in zip(fams, type_colors):
                    xs, st = bound_series(fam, greedy, f)
                    if not xs:
                        continue
                    lbl = family_title(fam).split("(")[1].rstrip(")")
                    errline(ax, xs, st, {"color": col, "ls": "-", "marker": "o"}, lbl)
                    if first is None or st[0][0] > first[1]:
                        first = (xs[0], st[0][0], xs[-1])
                if key == "a_iterations" and first:
                    n0, y0, n1 = first
                    xx = np.linspace(n0, n1, 200)
                    ax.plot(
                        xx,
                        y0 * np.sqrt(xx / n0),
                        color=pp.GREY,
                        ls="--",
                        lw=2.5,
                        label="sqrt(n) growth",
                    )
                finish(
                    fig,
                    ax,
                    f"bound_{kind}_{key}_greedy_{greedy}",
                    "Graph size [nodes]",
                    ylabel,
                )

    # 3. Worst case from the existing chains data (authors' edge-scan start).
    f5_path = RESULTS / "stageB_f5_edgescan.csv"
    if f5_path.exists():
        with open(f5_path, newline="") as fh:
            f5 = [
                (int(r["n"]), int(r["m"]), int(r["iterations"]), int(r["total_ops"]))
                for r in csv.DictReader(fh)
            ]
        f5.sort()
        xs = [math.sqrt(n) for n, *_ in f5]
        ys = [it for _n, _m, it, _o in f5]
        a, b = np.polyfit(xs, ys, 1)
        fig, ax = pp._new_chart(
            "Worst case: iterations grow like sqrt(n)",
            "Short-and-long chains from the authors' greedy start",
        )
        ax.plot(
            [0, max(xs) * 1.05],
            [b, a * max(xs) * 1.05 + b],
            color=pp.GREY,
            ls="--",
            lw=2.5,
            label=f"fitted line: {a:.2f} * sqrt(n) {b:+.1f}",
        )
        pp._line(
            ax, xs, ys, {"color": pp.BLUE, "ls": "-", "marker": "o", "label": "Gabow"}
        )
        finish(fig, ax, "worst_case_iterations_vs_sqrt_n", "sqrt(n)", "Iterations")
        fig, ax = pp._new_chart(
            "Worst case: total work / (sqrt(n) * m)",
            "Short-and-long chains from the authors' greedy start; flat = bound attained",
        )
        pp._line(
            ax,
            [n for n, *_ in f5],
            [o / (math.sqrt(n) * m) for n, m, _it, o in f5],
            {"color": pp.BLUE, "ls": "-", "marker": "o", "label": "Gabow"},
        )
        finish(
            fig,
            ax,
            "worst_case_work_bound",
            "Graph size [nodes]",
            "Operations / (sqrt(n) * m)",
        )

    # 4. Summary table.
    def rng(s, fmt):
        if not s:
            return ""
        return f"{fmt(s[0])} [{fmt(s[1])}-{fmt(s[2])}]"

    rows = []
    for fam in families:
        for n in sizes[fam]:
            t = {
                a: per_graph(times, fam, n, a)
                for a in ("edmonds", "gabow_on", "gabow_off")
            }
            o = {g: ops_per_graph(ops, fam, n, g) for g in ("on", "off")}
            row = {
                "family": fam,
                "n": n,
                "m_median": statistics.median(m_of[fam, n]),
                "graphs": len(o["on"]),
            }
            for a in ("edmonds", "gabow_on", "gabow_off"):
                row[f"time_{a}"] = _stats(t[a].values())
            for g in ("on", "off"):
                sp = [
                    t["edmonds"][s] / t[f"gabow_{g}"][s]
                    for s in t["edmonds"]
                    if s in t[f"gabow_{g}"]
                ]
                row[f"speedup_{g}"] = _stats(sp)
                row[f"iterations_{g}"] = _stats(
                    [r["iterations"] for r in o[g].values()]
                )
                row[f"ops_per_iter_m_{g}"] = _stats(
                    [r["total_ops"] / r["iterations"] / r["m"] for r in o[g].values()]
                )
                row[f"ops_bound_{g}"] = _stats(
                    [
                        r["total_ops"] / (math.sqrt(r["n"]) * r["m"])
                        for r in o[g].values()
                    ]
                )
            rows.append(row)
    stat_keys = [k for k in rows[0] if k not in ("family", "n", "m_median", "graphs")]
    with open(outdir / "summary.csv", "w", newline="") as fh:
        fields = ["family", "n", "m_median", "graphs"]
        for k in stat_keys:
            fields += [f"{k}_median", f"{k}_min", f"{k}_max"]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            out = {k: r[k] for k in ("family", "n", "m_median", "graphs")}
            for k in stat_keys:
                s = r[k]
                out[f"{k}_median"], out[f"{k}_min"], out[f"{k}_max"] = (
                    s[:3] if s else ("", "", "")
                )
            w.writerow(out)
    sec = lambda v: f"{v:.3g}"
    x = lambda v: f"{v:.3g}x"
    g3 = lambda v: f"{v:.3g}"
    md = [
        "# Final runs: summary\n",
        (
            "Median over the graphs of each (family, n) with [min-max] over graphs."
            " Times in seconds (counters off); speedup = Edmonds / Gabow on the same"
            " graph; ops from a separate run with counters on.\n"
        ),
        (
            "| family | n | m | graphs | Edmonds s | Gabow ON s | Gabow OFF s"
            " | speedup ON | speedup OFF | iter ON | iter OFF | ops/iter/m OFF"
            " | ops/(sqrt(n) m) OFF |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md.append(
            f"| {family_title(r['family'])} | {r['n']:,} | {r['m_median']:,.0f}"
            f" | {r['graphs']} | {rng(r['time_edmonds'], sec)}"
            f" | {rng(r['time_gabow_on'], sec)} | {rng(r['time_gabow_off'], sec)}"
            f" | {rng(r['speedup_on'], x)} | {rng(r['speedup_off'], x)}"
            f" | {rng(r['iterations_on'], g3)} | {rng(r['iterations_off'], g3)}"
            f" | {rng(r['ops_per_iter_m_off'], g3)} | {rng(r['ops_bound_off'], g3)} |"
        )
    (outdir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"charts written to {charts}")


# --------------------------------------------------------------------------


FULL = {
    "outdir": RESULTS / "final",
    "seeds": list(SEEDS),
    "sparse_cs": SPARSE_CS,
    "sparse_sizes": SPARSE_SIZES,
    "sparse_gabow_only": SPARSE_GABOW_ONLY,
    "sparse_late": 2,  # 15,000 and 20,000 run last
    "dense_ps": DENSE_PS,
    "dense_sizes": DENSE_SIZES,
    "edmonds_timeout": EDMONDS_TIMEOUT,
}
TINY = {
    "outdir": RESULTS / "final_tiny",
    "seeds": [0, 1],
    "sparse_cs": (3, 10),
    "sparse_sizes": (100, 200, 400),
    "sparse_gabow_only": (2000,),
    "sparse_late": 1,
    "dense_ps": (0.5, 1.0),
    "dense_sizes": (30, 60, 120),
    "edmonds_timeout": 0.05,  # forces timeouts, to test the skip logic
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tiny", action="store_true", help="small smoke test")
    p.add_argument("--charts-only", action="store_true")
    p.add_argument("--plan", action="store_true", help="print the run order")
    args = p.parse_args(argv)
    cfg = TINY if args.tiny else FULL
    if args.plan:
        for step in plan(cfg):
            print(*step)
        return
    if not args.charts_only:
        store = Store(cfg["outdir"])
        log(
            f"start; {len(store.times)} time rows and {len(store.ops)} ops rows on disk"
        )
        run(cfg, store)
    make_charts(cfg["outdir"])


if __name__ == "__main__":
    main()
