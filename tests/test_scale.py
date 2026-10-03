"""Runs the 100k-record scale benchmark in a clean subprocess and checks its
reported pair count, CPU time, and peak RSS."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent / "scale_benchmark.py"


def test_scale_benchmark():
    proc = subprocess.run(
        [sys.executable, str(BENCH)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    print(proc.stdout)
    print(proc.stderr, file=sys.stderr)
    assert proc.returncode == 0, proc.stderr
    assert "SCALE BENCHMARK OK" in proc.stdout
