"""Checks for the Phase 6 proxy reconstruction study."""

from __future__ import annotations

import numpy as np

from km3disp.geometry import NominalGeometry
from km3disp.study import (
    cherenkov_hit_times,
    line_fit_direction,
    opening_angle_deg,
    proxy_study,
)


def test_line_fit_recovers_clean_track():
    # Hits sampled along a track, Cherenkov times, no displacement/noise: the line
    # fit should point roughly along the track (proxy-level accuracy).
    rng = np.random.default_rng(1)
    d = np.array([0.3, 0.4, np.sqrt(1 - 0.25)])
    d /= np.linalg.norm(d)
    p0 = np.array([0.0, 0.0, -3200.0])
    pos = p0 + rng.uniform(-300, 300, (60, 1)) * d + rng.uniform(-40, 40, (60, 3))
    t = cherenkov_hit_times(p0, d, pos)
    assert opening_angle_deg(line_fit_direction(pos, t), d) < 10.0


def test_calibration_lag_degrades_with_displacement():
    geo = NominalGeometry.load("arca")
    res = proxy_study(geo, [0.0, 0.05, 0.15], n_events=150, seed=0, solve_nodes=601)
    b0 = res[0.0]["mode_b_median_deg"]
    b_hi = res[0.15]["mode_b_median_deg"]
    a0 = res[0.0]["mode_a_median_deg"]
    a_hi = res[0.15]["mode_a_median_deg"]
    # mode (b) clearly worsens with displacement ...
    assert b_hi > b0 + 0.5, (b0, b_hi)
    # ... while mode (a) stays close to baseline (intrinsic effect is small)
    assert abs(a_hi - a0) < b_hi - b0, (a0, a_hi, b0, b_hi)
    assert res[0.15]["max_displacement_m"] > 50.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
