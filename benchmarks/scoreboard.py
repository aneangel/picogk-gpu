"""Aggregate benchmark JSON files in results/ into a human-readable scoreboard.

Usage:
  python benchmarks/scoreboard.py
  python benchmarks/scoreboard.py --results-dir benchmarks/results
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_results(results_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(results_dir.glob("*.json"))]


def print_scoreboard(results: list[dict]) -> None:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in results:
        groups[(r["name"], r.get("voxel_size"))].append(r)

    header = f"{'benchmark':<22} {'voxel':>8} {'wall(s)':>12} {'rss(MiB)':>10} {'runs':>5} {'when':<19}"
    print(header)
    print("-" * len(header))

    for (name, vox), recs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        recs.sort(key=lambda r: r["timestamp"], reverse=True)
        latest = recs[0]
        agg = latest.get("aggregate", {})
        wall = agg.get("wall_seconds", {}).get("median", float("nan"))
        rss_kb = agg.get("max_rss_kb", {}).get("median", float("nan"))
        rss_mib = (rss_kb / 1024) if isinstance(rss_kb, (int, float)) else float("nan")
        n = agg.get("wall_seconds", {}).get("n", 0)
        vox_str = f"{vox}" if vox is not None else "-"
        print(f"{name:<22} {vox_str:>8} {wall:>12.3f} {rss_mib:>10.1f} {n:>5} {latest['timestamp']:<19}")


def _cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(Path(__file__).parent / "results"))
    a = parser.parse_args()
    results = load_results(Path(a.results_dir))
    if not results:
        print(f"no results in {a.results_dir}")
        return
    print_scoreboard(results)


if __name__ == "__main__":
    _cli()
