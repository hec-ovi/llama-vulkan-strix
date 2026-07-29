"""Unit tests for scripts/bench_full.py (no server required)."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "bench_full", ROOT / "scripts" / "bench_full.py"
)
mod = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(mod)


def test_build_prompt_scales_with_depth():
    short = mod.build_prompt(64, 0)
    long = mod.build_prompt(16384, 0)
    assert len(long) > len(short)
    assert "Benchmark run" in short


def test_stats_mean_median():
    s = mod.stats([10.0, 20.0, 30.0])
    assert s["n"] == 3
    assert s["mean"] == 20.0
    assert s["median"] == 20.0
    assert s["min"] == 10.0
    assert s["max"] == 30.0


def test_stats_empty():
    s = mod.stats([])
    assert s["n"] == 0
    assert s["mean"] == 0.0
