"""Empirical verification of max_cardinality_matching_gabow's claimed
O(sqrt(n) * m) time bound, using the `_counters`/`_skip_greedy_init`
instrumentation hooks on max_cardinality_matching_gabow (additive-only;
see that function's docstring -- no algorithm logic or results change).

Standalone script, not collected by pytest, writes new CSV/PNG files under
benchmarks/results/ (does not touch the files from bench_matching.py or
gabow_complexity_check.py):

    python benchmarks/verify_complexity.py

bound(n, m) = sqrt(n) * m throughout, per Theorem 5.1 of Gabow (2017) /
Section 4 of Mehlhorn & Nobahari's "Revisited" paper (arXiv:2603.22909),
which states the per-iteration union-find cost as O(m) or O(m*alpha(n))
"depending on the sophistication of the realization" -- see the code-audit
summary printed at the end and in the conversation this script accompanies.

total_ops is the sum of: edge_scans, search_steps,
blossom_shrink_vertex_steps, uf_*_find_hops, uf_*_union_calls, and
augmentations -- the counters that represent O(1)-ish elementary work
units (as opposed to blossom_contraction_events/iterations/delta_phases,
which are *counts of events*, not units of work). total_ops_excl_uf
subtracts the uf_*_find_hops/union_calls terms, isolating whether any
superlinear growth is coming specifically from the union-find structure.
"""

import csv
import math
import time
import warnings
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import networkx as nx

OUTDIR = Path(__file__).with_name("results")
OUTDIR.mkdir(parents=True, exist_ok=True)

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
UF_KEYS = [
    "uf_base_find_hops",
    "uf_dbase_find_hops",
    "uf_base_union_calls",
    "uf_dbase_union_calls",
]


def bound(n, m):
    return math.sqrt(n) * m


def _safe_ratios(values):
    """values[i]/values[i-1] for consecutive pairs; None where the
    denominator is 0 (a legitimate outcome here -- e.g. the greedy
    warm-start already found a perfect matching, so Phase 1 never had a
    free vertex to scan from, giving edge_scans=0 -- rather than a bug).
    """
    out = []
    for i in range(1, len(values)):
        prev = values[i - 1]
        out.append(values[i] / prev if prev else None)
    return out


def measure(G, skip_greedy=False):
    c = {}
    t0 = time.perf_counter()
    matching = nx.max_cardinality_matching_gabow(
        G, _counters=c, _skip_greedy_init=skip_greedy
    )
    elapsed = time.perf_counter() - t0
    total_ops = sum(c[k] for k in OP_KEYS)
    uf_ops = sum(c[k] for k in UF_KEYS)
    row = {
        "n": G.number_of_nodes(),
        "m": G.number_of_edges(),
        "time": elapsed,
        "matching_size": len(matching),
        "total_ops": total_ops,
        "total_ops_excl_uf": total_ops - uf_ops,
        "uf_ops": uf_ops,
        "iterations": c["iterations"],
        "delta_phases": c["delta_phases"],
        "blossom_contraction_events": c["blossom_contraction_events"],
        "blossom_expansions": c["blossom_expansions"],
    }
    for k in OP_KEYS:
        row[k] = c[k]
    return row


def _fit_loglog(xs, ys):
    # Drop non-positive y (e.g. ops=0 when the greedy warm-start already
    # found a perfect matching, so Phase 1 had nothing to scan) -- a
    # legitimate outcome, but log(0) is undefined and would otherwise NaN
    # out the whole fit.
    pairs = [(x, y) for x, y in zip(xs, ys) if y > 0]
    xs, ys = zip(*pairs) if pairs else ([], [])
    lx = np.log(np.asarray(xs, dtype=float))
    ly = np.log(np.asarray(ys, dtype=float))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.exceptions.RankWarning)
        slope, intercept = np.polyfit(lx, ly, 1)
    fitted = slope * lx + intercept
    ssr = float(np.sum((ly - fitted) ** 2))
    sst = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1 - ssr / sst if sst > 0 else float("nan")
    npts = len(lx)
    sxx = float(np.sum((lx - lx.mean()) ** 2))
    se = float(np.sqrt((ssr / (npts - 2)) / sxx)) if npts > 2 and sxx > 0 else float("nan")
    return float(slope), float(intercept), r2, se


def _fit_multiple(ns, ms, ys):
    """log(y) = a*log(n) + b*log(m) + c, via least squares. Returns
    (a, b, c, se_a, se_b).
    """
    X = np.column_stack(
        [np.log(np.asarray(ns, float)), np.log(np.asarray(ms, float)), np.ones(len(ns))]
    )
    y = np.log(np.asarray(ys, float))
    coef, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ coef
    resid = y - fitted
    n_pts, n_params = X.shape
    dof = max(n_pts - n_params, 1)
    sigma2 = float(np.sum(resid**2)) / dof
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    return float(coef[0]), float(coef[1]), float(coef[2]), float(se[0]), float(se[1])


def write_csv(rows, name, fieldnames):
    path = OUTDIR / name
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return path


def plot_ratio(rows, xkey, outfile, title):
    ns = [r["n"] for r in rows]
    ops_ratio = [r["total_ops"] / bound(r["n"], r["m"]) for r in rows]
    ops_excl_ratio = [r["total_ops_excl_uf"] / bound(r["n"], r["m"]) for r in rows]
    time_ratio = [r["time"] / bound(r["n"], r["m"]) for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(ns, ops_ratio, "o-", label="total_ops / bound(n,m)")
    ax1.plot(ns, ops_excl_ratio, "s--", label="total_ops_excl_uf / bound(n,m)")
    ax1.set_xlabel(xkey)
    ax1.set_ylabel("ops / (sqrt(n)*m)")
    ax1.set_title(f"{title}: ops ratio")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.plot(ns, time_ratio, "o-", color="tab:green")
    ax2.set_xlabel(xkey)
    ax2.set_ylabel("time / (sqrt(n)*m)")
    ax2.set_title(f"{title}: time ratio")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTDIR / outfile, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Graph families
# ---------------------------------------------------------------------------


def complete_g(m_target):
    """Clique on n=int(sqrt(m_target)) vertices; returns (G, z, next_id)."""
    n = max(2, int(math.sqrt(m_target)))
    G = nx.Graph()
    nodes = list(range(n))
    G.add_nodes_from(nodes)
    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(nodes[i], nodes[j])
    return G, nodes[0], n


def _add_chain(G, k, z, next_id):
    """Direct Python translation of chain(G,k,z) from the companion-page
    C++ reference (mc_timing_{short,long}_chains.cpp): a k-node chain
    attached to z, where every interior vertex *also* has a direct edge to
    z (not merely a simple path) -- this is the authors' exact construction,
    not an approximation of it.
    """
    a = list(range(next_id, next_id + k))
    G.add_nodes_from(a)
    G.add_edge(a[0], z)
    for i in range(1, k - 1):
        G.add_edge(a[i], a[i + 1])
        G.add_edge(a[i], z)
    G.add_edge(a[0], a[1])
    return next_id + k


def chains_graph(n_param, mode=1):
    """Python port of mc_worst_case_gen(n, m, mode) from the companion-page
    C++ reference, with m_param = 4*n_param as in the authors' own
    experiment drivers. mode=0: "short chains only" (n_param/8 chains of 8
    nodes each). mode=1 (default): the same short chains, plus one chain
    of length 2k for each 5 <= k < sqrt(n_param) ("short and long chains").
    """
    m_param = 4 * n_param
    G, z, _clique_n = complete_g(m_param)
    next_id = G.number_of_nodes()
    for _j in range(n_param // 8):
        next_id = _add_chain(G, 8, z, next_id)
    if mode == 1:
        k = 5
        while k < math.sqrt(n_param):
            next_id = _add_chain(G, 2 * k, z, next_id)
            k += 1
    return G


def bad_greedy_graph(k):
    """k disjoint copies of a 4-node gadget that defeats this specific
    implementation's greedy warm-start (`for v in G: match v to its first
    unmatched neighbor`): with nodes inserted in the order 2,3,1,4 and
    edges (2,3),(1,2),(3,4), greedy visits node 2 first and has neighbors
    [3,1] in that order, so it grabs the *middle* edge 2-3 first, stranding
    1 and 4 -- leaving an augmenting path 1-2-3-4 that must be found by the
    real search. Verified directly (see conversation): this specific
    4-node graph provably forces one real augmentation past greedy.
    """
    G = nx.Graph()
    for i in range(k):
        b = 4 * i
        G.add_edge(b + 2, b + 3)
        G.add_edge(b + 1, b + 2)
        G.add_edge(b + 3, b + 4)
    return G


def nested_blossoms_graph(depth):
    """A chain of `depth` triangles sharing tips, as in
    TestMaxCardinalityMatchingGabow._nested_blossoms -- forces `depth`
    levels of blossom contraction in one search.
    """
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    tip = 0
    next_id = 3
    for _ in range(depth - 1):
        a, b = next_id, next_id + 1
        G.add_edges_from([(tip, a), (a, b), (b, tip)])
        tip = a
        next_id += 2
    return G


# ---------------------------------------------------------------------------
# Experiment (a): fixed n, varying m
# ---------------------------------------------------------------------------


def experiment_a():
    print("\n=== Experiment (a): fixed n, varying m ===")
    n = 600
    ms = [int(n * c) for c in (2, 4, 8, 16, 32, 48, 64)]  # up to near-complete
    rows = []
    for m in ms:
        m = min(m, n * (n - 1) // 2)
        G = nx.gnm_random_graph(n, m, seed=1)
        row = measure(G)
        rows.append(row)
        print(f"  n={n} m={row['m']:6d} ops={row['total_ops']:8d} time={row['time']:.4f}s")
    write_csv(
        rows,
        "verify_a_fixed_n_varying_m.csv",
        list(rows[0].keys()),
    )
    # ops should grow ~linearly in m at fixed n: fit log(ops) vs log(m) only
    ms_actual = [r["m"] for r in rows]
    ops = [r["total_ops"] for r in rows]
    slope, _intercept, r2, se = _fit_loglog(ms_actual, ops)
    plot_ratio(rows, "m", "verify_a_fixed_n_varying_m.png", f"(a) fixed n={n}, varying m")
    return rows, {"slope_vs_m": slope, "se": se, "r2": r2}


# ---------------------------------------------------------------------------
# Experiment (b): m = c*n for several c, varying n
# ---------------------------------------------------------------------------


def experiment_b(max_n):
    print("\n=== Experiment (b): m = c*n, varying n ===")
    results = {}
    sizes = [n for n in (100, 200, 400, 800, 1600, 3200, 6400) if n <= max_n]
    for c in (2, 4, 8):
        rows = []
        for n in sizes:
            m = c * n
            G = nx.gnm_random_graph(n, m, seed=2)
            row = measure(G)
            rows.append(row)
        write_csv(rows, f"verify_b_mc{c}n.csv", list(rows[0].keys()))
        ns = [r["n"] for r in rows]
        ops = [r["total_ops"] for r in rows]
        slope, _i, r2, se = _fit_loglog(ns, ops)
        plot_ratio(rows, "n", f"verify_b_mc{c}n.png", f"(b) m={c}*n")
        print(f"  c={c}: ops ~ n^{slope:.2f} (se={se:.2f}, R2={r2:.3f})")
        results[c] = {"rows": rows, "slope_vs_n": slope, "se": se, "r2": r2}
    return results


# ---------------------------------------------------------------------------
# Experiment (c): fixed densities, varying n
# ---------------------------------------------------------------------------


def experiment_c(max_n):
    print("\n=== Experiment (c): fixed densities, varying n ===")
    # NOTE: at fixed density m is a deterministic function of n
    # (m = d*n*(n-1)/2), so log(n) and log(m) are collinear here -- a
    # two-variable regression would be ill-posed (see experiment_combined_ab
    # for the genuine two-variable fit, which instead pools experiments (a)
    # and (b), where n and m vary independently). Here we only fit the
    # single-variable slope of ops vs n, which is what "flat/decreasing/
    # increasing ratio per density" actually asks for.
    densities = [0.10, 0.25, 0.50, 0.75, 1.00]
    sizes = [n for n in (50, 100, 150, 200, 300, 400, 600) if n <= max_n]
    results = {}
    for d in densities:
        rows = []
        for n in sizes:
            m = round(d * n * (n - 1) / 2)
            G = nx.gnm_random_graph(n, m, seed=3)
            rows.append(measure(G))
        write_csv(rows, f"verify_c_density{round(d*100)}.csv", list(rows[0].keys()))
        ns = [r["n"] for r in rows]
        ops = [r["total_ops"] for r in rows]
        plot_ratio(rows, "n", f"verify_c_density{round(d*100)}.png", f"(c) density={round(d*100)}%")
        if any(o > 0 for o in ops):
            slope, _i, r2, se = _fit_loglog(ns, ops)
            print(f"  density={d:.2f}: ops ~ n^{slope:.2f} (se={se:.2f}, R2={r2:.3f})")
            results[d] = {"rows": rows, "slope_vs_n": slope, "se": se, "r2": r2}
        else:
            # Complete graphs (density 100%): the greedy warm-start alone
            # (processing nodes in order, every unmatched node adjacent to
            # every other) always finds the perfect matching directly, so
            # Phase 1/2 never run -- ops=0 at every size, a genuine result
            # (trivial instance), not a fit failure.
            print(f"  density={d:.2f}: ops=0 at every size -- greedy alone solves K_n")
            results[d] = {"rows": rows, "slope_vs_n": None, "se": None, "r2": None}
    return results


def experiment_combined_ab(a_rows, b_results):
    """The genuine two-variable regression log(ops) = a*log(n) + b*log(m) + c,
    using experiment (a) [fixed n=600, varying m] pooled with experiment (b)
    [m=c*n for c in {2,4,8}, varying n] -- here n and m actually vary
    independently (unlike experiment (c)'s fixed-density sweep), so the fit
    is well-posed.
    """
    rows = list(a_rows)
    for res in b_results.values():
        rows.extend(res["rows"])
    ns = [r["n"] for r in rows]
    ms = [r["m"] for r in rows]
    ops = [r["total_ops"] for r in rows]
    a, b, c0, se_a, se_b = _fit_multiple(ns, ms, ops)
    print(
        f"\n=== Combined (a)+(b) two-variable fit (n, m vary independently) ===\n"
        f"  log(ops) ~ {a:.3f}*log(n) + {b:.3f}*log(m) + {c0:.3f}\n"
        f"  se(a)={se_a:.3f}  se(b)={se_b:.3f}\n"
        f"  expected upper bounds: a <= 0.5, b <= 1"
    )
    return {"a": a, "b": b, "c0": c0, "se_a": se_a, "se_b": se_b}


# ---------------------------------------------------------------------------
# Experiment (d): adversarial families vs random
# ---------------------------------------------------------------------------


def experiment_d(max_chain_n):
    print("\n=== Experiment (d): adversarial families vs random ===")
    all_rows = {}

    # (d1) chains families, from the authors' own construction, run with
    # the algorithm's normal (greedy-included) initialization AND with
    # the greedy warm-start skipped (testing Fig. 1's bare algorithm,
    # which is what the O(sqrt(n)) phase-count claim is actually about).
    for mode, label in [(0, "short_only"), (1, "short_and_long")]:
        for skip in (False, True):
            rows = []
            n_param = 500
            while n_param <= max_chain_n:
                G = chains_graph(n_param, mode=mode)
                row = measure(G, skip_greedy=skip)
                row["n_param"] = n_param
                rows.append(row)
                n_param *= 2
            tag = f"{label}_{'skipgreedy' if skip else 'greedy'}"
            write_csv(rows, f"verify_d_{tag}.csv", list(rows[0].keys()))
            all_rows[tag] = rows
            ratios = _safe_ratios([r["edge_scans"] for r in rows])
            ratios_fmt = [round(x, 2) if x is not None else None for x in ratios]
            print(
                f"  {tag}: n from {rows[0]['n']} to {rows[-1]['n']}, "
                f"iterations={[r['iterations'] for r in rows]}, "
                f"edge_scans doubling ratios={ratios_fmt}"
            )

    # (d2) bad-greedy-start family, at matched sizes to (d3) random.
    rows_bg = []
    for k in (50, 100, 200, 400, 800, 1600):
        G = bad_greedy_graph(k)
        rows_bg.append(measure(G))
    write_csv(rows_bg, "verify_d_bad_greedy.csv", list(rows_bg[0].keys()))
    all_rows["bad_greedy"] = rows_bg
    print(f"  bad_greedy: augmentations={[r['augmentations'] for r in rows_bg]} "
          f"(size k => k augmentations expected if greedy is defeated every block)")

    # (d3) nested blossoms.
    rows_nb = []
    for depth in (5, 10, 20, 40, 80, 160):
        G = nested_blossoms_graph(depth)
        rows_nb.append(measure(G))
    write_csv(rows_nb, "verify_d_nested_blossoms.csv", list(rows_nb[0].keys()))
    all_rows["nested_blossoms"] = rows_nb
    print(f"  nested_blossoms: blossom_contraction_events={[r['blossom_contraction_events'] for r in rows_nb]}")

    # (d4) random graphs at the same (n, m) as the short_and_long family,
    # for direct comparison (same n and m, different structure).
    rows_rand = []
    for r in all_rows["short_and_long_greedy"]:
        G = nx.gnm_random_graph(r["n"], r["m"], seed=4)
        rows_rand.append(measure(G))
    write_csv(rows_rand, "verify_d_random_matched.csv", list(rows_rand[0].keys()))
    all_rows["random_matched"] = rows_rand
    ratios_rand = _safe_ratios([r["edge_scans"] for r in rows_rand])
    ratios_rand_fmt = [round(x, 2) if x is not None else None for x in ratios_rand]
    print(
        f"  random_matched: iterations={[r['iterations'] for r in rows_rand]}, "
        f"edge_scans doubling ratios={ratios_rand_fmt}"
    )

    # combined ops-ratio plot across all (d) families
    fig, ax = plt.subplots(figsize=(8, 5))
    for tag, rows in all_rows.items():
        ns = [r["n"] for r in rows]
        ratio = [r["total_ops"] / bound(r["n"], r["m"]) for r in rows]
        ax.plot(ns, ratio, "o-", label=tag)
    ax.set_xlabel("n")
    ax.set_ylabel("total_ops / (sqrt(n)*m)")
    ax.set_title("(d) adversarial vs. random: ops ratio")
    ax.set_xscale("log")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTDIR / "verify_d_combined_ratio.png", dpi=150)
    plt.close(fig)

    return all_rows


if __name__ == "__main__":
    total_t0 = time.perf_counter()

    a_rows, a_fit = experiment_a()
    b_results = experiment_b(max_n=6400)
    combined_ab_fit = experiment_combined_ab(a_rows, b_results)
    c_results = experiment_c(max_n=600)
    d_results = experiment_d(max_chain_n=16000)

    total_t1 = time.perf_counter()
    print(f"\nTotal run time: {total_t1 - total_t0:.1f}s")
