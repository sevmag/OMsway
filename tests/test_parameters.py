"""Regression checks for ARCA/ANTARES parameters and calibration."""

from __future__ import annotations

from km3disp.geometry import NominalGeometry
from km3disp.mechanics import uniform_current
from km3disp import parameters as P


def test_antares_matches_published_deflections():
    # Published: top storey < 2 m at 7 cm/s, ~15 m at 20 cm/s (arXiv:1202.3894).
    model = P.antares_model()
    assert model.solve(uniform_current(0.07)).top_dom_deflection < 2.0
    assert 5.0 < model.solve(uniform_current(0.20)).top_dom_deflection < 25.0


def test_arca_calibration_hits_benchmark():
    scale = P.calibrate_arca()
    assert 0.1 < scale < 5.0  # benign, physically reasonable adjustment
    consts = P.arca_constants(drag_scale=scale)
    shape = P.benchmark_du_model(consts).solve(uniform_current(0.15))
    assert abs(shape.buoy_deflection - 100.0) < 1.0


def test_arca_realistic_regime():
    scale = P.calibrate_arca()
    geo = NominalGeometry.load("arca")
    model = P.arca_du_model(geo.units[0], drag_scale=scale)
    # Typical ARCA current (2-3 cm/s) -> a few metres at the top DOM.
    assert 1.0 < model.solve(uniform_current(0.03)).top_dom_deflection < 6.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
