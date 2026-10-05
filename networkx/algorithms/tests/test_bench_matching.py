"""End-to-end smoke test for benchmarks/bench_matching.py.

Loads that standalone script as a module (it lives outside the networkx
package, under benchmarks/, so it is loaded by file path rather than
imported normally) and runs its pipeline at a tiny, fast size -- not the
full benchmark configuration -- so this runs as part of the regular test
suite rather than needing @pytest.mark.slow / --runslow.
"""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("matplotlib")


def _load_bench_matching():
    path = (
        Path(__file__).resolve().parents[3] / "benchmarks" / "bench_matching.py"
    )
    if not path.exists():
        pytest.skip(f"benchmarks/bench_matching.py not found at {path}")
    spec = importlib.util.spec_from_file_location("bench_matching", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bench_matching_quick_pipeline(tmp_path):
    bench_matching = _load_bench_matching()

    sizes = [12, 20, 28]
    densities = [0.3]
    repeats = 2

    # run_experiments() asserts internally that the two algorithms agree on
    # matching cardinality for every generated graph, raising loudly if not.
    rows = bench_matching.run_experiments(sizes, densities, repeats)
    assert len(rows) == len(sizes) * len(densities) * repeats * len(
        bench_matching.ALGORITHMS
    )

    seen = {}
    for row in rows:
        key = (row["density"], row["n"], row["seed"])
        seen.setdefault(key, set()).add(row["algorithm"])
    for algos_present in seen.values():
        assert algos_present == set(bench_matching.ALGORITHMS)

    csv_path = bench_matching.write_csv(rows, tmp_path)
    assert csv_path.exists()
    assert csv_path.read_text().splitlines()[0] == (
        "density,n,m,seed,algorithm,runtime_sec"
    )

    summary_csv_path = bench_matching.write_summary_csv(
        rows, densities, sizes, tmp_path
    )
    assert summary_csv_path.exists()
    assert summary_csv_path.read_text().splitlines()[0] == (
        "density,n,algorithm,mean,median,std,repetitions"
    )
    # one summary row per (density, n, algorithm) combination
    assert len(summary_csv_path.read_text().splitlines()) == 1 + len(
        densities
    ) * len(sizes) * len(bench_matching.ALGORITHMS)

    outpath, fig_summary = bench_matching.make_plot(
        rows, densities[0], sizes, tmp_path, loglog_inset=True
    )
    assert outpath.exists()
    assert outpath.stat().st_size > 0
    assert set(fig_summary) == set(bench_matching.ALGORITHMS)
    for algo_summary in fig_summary.values():
        assert "slope" in algo_summary
        assert "r2" in algo_summary
        assert "se_slope" in algo_summary


def test_bench_matching_main_quick_smoke(tmp_path):
    bench_matching = _load_bench_matching()
    rows, summaries = bench_matching.main(
        [
            "--quick",
            "--sizes",
            "12",
            "20",
            "--densities",
            "0.4",
            "--repeats",
            "2",
            "--outdir",
            str(tmp_path),
            "--no-loglog-inset",
        ]
    )
    assert rows
    assert (tmp_path / "raw_results.csv").exists()
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "runtime_density_40.png").exists()
    assert 0.4 in summaries
