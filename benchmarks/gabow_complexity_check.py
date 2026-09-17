"""Standalone complexity sanity-check for max_cardinality_matching_gabow.

This is NOT an asv benchmark (see benchmarks/benchmarks/ and README.md for
those) and it is NOT collected by pytest. It is a plain script you run
directly:

    python benchmarks/gabow_complexity_check.py

It times ``networkx.max_cardinality_matching_gabow(G,
use_heuristic_fallback=False)`` -- deliberately with the heuristic fallback
off, so it measures Gabow's algorithm itself (Fig. 1 of the paper) rather
than the constant-factor practical shortcut -- on sparse random graphs of
increasing size, and checks whether the measured growth is consistent with
the paper's O(sqrt(n) * m) time bound (Theorem 5.1 of
"The Weighted Matching Approach to Maximum Cardinality Matching", H.N.
Gabow, Fundamenta Informaticae 154 (2017)).

Methodology
-----------
Sparse random graphs with a fixed average degree (m = density * n) are used
so that n is the only free variable and the sqrt(n) * m bound reduces to a
single power law, ~ n**1.5, that is easy to fit on a log-log plot.

This is a *sanity check*, not a worst-case stress test: it does not
construct the adversarial graph families that are known to force the full
Theta(sqrt(n)) phase count (those require deliberately engineered
bipartite-like gadgets). So passing this check does not by itself *prove*
the bound is tight or even attained here -- only that nothing asymptotically
worse than claimed is happening on ordinary sparse inputs. Failing it,
conversely, is a real signal: it means some operation the implementation
assumes is O(1) amortized (a union-find step, a bucket-queue insert, an
edge scan) has regressed into something that scales with n or m instead.
"""

import math
import time

import networkx as nx


def _time_once(G):
    t0 = time.perf_counter()
    nx.max_cardinality_matching_gabow(G, use_heuristic_fallback=False)
    return time.perf_counter() - t0


def _median_time(n, m, repeats, base_seed):
    times = sorted(
        _time_once(nx.gnm_random_graph(n, m, seed=base_seed + i))
        for i in range(repeats)
    )
    return times[len(times) // 2]


def _least_squares_slope(xs, ys):
    """Ordinary least-squares slope of ys against xs (both pre-logged)."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    return num / den


def main():
    sizes = [100, 200, 400, 800, 1600, 3200, 6400]
    density = 3  # m = density * n edges (average degree ~= 2 * density)
    repeats = 3

    rows = []
    print(f"{'n':>8}{'m':>10}{'median time (s)':>18}{'t / (sqrt(n)*m)':>20}")
    for n in sizes:
        m = density * n
        t = _median_time(n, m, repeats, base_seed=n)
        model = math.sqrt(n) * m
        rows.append((n, m, t, model))
        print(f"{n:>8}{m:>10}{t:>18.5f}{t / model:>20.3e}")

    log_n = [math.log(n) for n, _, _, _ in rows]
    log_t = [math.log(t) for _, _, t, _ in rows if t > 0]
    if len(log_t) != len(log_n):
        # Guard against a timing that rounded to 0.0 on a very fast run.
        log_n = log_n[-len(log_t) :]

    empirical_exponent = _least_squares_slope(log_n, log_t)
    # sqrt(n) * m with m = density * n is Theta(n**1.5); holding average
    # degree fixed makes n the only free variable, so the bound predicts a
    # slope of about 1.5 on this log-log plot.
    expected_exponent = 1.5

    print()
    print(f"Empirical time ~ n^{empirical_exponent:.2f}")
    print(
        f"O(sqrt(n)*m) bound with m = {density}*n predicts ~ n^{expected_exponent:.2f}"
    )

    # A generous tolerance: real measurements are noisy (GC, cache effects,
    # small sizes not yet in the asymptotic regime, the log-factor from
    # using path-compression-only union-find noted in the function's
    # docstring). What this is really watching for is drifting toward
    # n**2.5 (the profile of doing an O(n)-per-augmentation search m times
    # with no phase batching at all) or n**3 (the profile of the
    # Galil/Edmonds max_weight_matching baseline) -- either would mean a
    # full extra power of n crept in somewhere.
    tolerance = 0.6
    print()
    if empirical_exponent > expected_exponent + tolerance:
        print("WARNING: measured growth rate is NOT consistent with the")
        print(
            f"claimed O(sqrt(n)*m) bound (exponent {empirical_exponent:.2f} vs."
        )
        print(f"expected ~{expected_exponent:.2f}). This suggests the")
        print("implementation has lost the complexity guarantee somewhere")
        print("(e.g. an accidental O(n) or O(m) scan inside what should be")
        print("an O(1)-amortized step).")
    else:
        print("Measured growth rate is consistent with the claimed")
        print("O(sqrt(n)*m) bound.")


if __name__ == "__main__":
    main()
