"""Checks for the depth-resolved current profiles."""

from __future__ import annotations

import numpy as np

from km3disp import currents as C
from km3disp import parameters as P
from km3disp.geometry import NominalGeometry


def test_uniform_is_constant():
    f = C.uniform(0.1, azimuth_deg=45.0)
    u = f(np.array([-3500.0, -3000.0, -2900.0]))
    assert np.allclose(u, u[0])
    assert abs(np.hypot(*u[0]) - 0.1) < 1e-9


def test_sheared_speed_varies_with_depth():
    p = C.sheared_speed(-3500.0, -2900.0, speed_bottom=0.02, speed_top=0.10)
    assert abs(np.hypot(*p(np.array([-3500.0]))[0]) - 0.02) < 1e-9
    assert abs(np.hypot(*p(np.array([-2900.0]))[0]) - 0.10) < 1e-9
    # direction unchanged
    assert abs(p.azimuth_deg.std()) < 1e-9


def test_edge_clamping():
    p = C.sheared_speed(-3500.0, -2900.0, 0.02, 0.10)
    # beyond the node range, hold the edge value (no extrapolation blow-up)
    assert abs(np.hypot(*p(np.array([-4000.0]))[0]) - 0.02) < 1e-9
    assert abs(np.hypot(*p(np.array([-2000.0]))[0]) - 0.10) < 1e-9


def test_rotating_azimuth_changes_direction():
    p = C.rotating_azimuth(-3500.0, -2900.0, speed=0.08,
                           azimuth_bottom_deg=0.0, azimuth_top_deg=90.0)
    bottom = p(np.array([-3500.0]))[0]
    top = p(np.array([-2900.0]))[0]
    assert abs(np.degrees(np.arctan2(bottom[1], bottom[0])) - 0.0) < 2.0
    assert abs(np.degrees(np.arctan2(top[1], top[0])) - 90.0) < 2.0


def test_benthic_reduces_near_bed():
    base = C.uniform(0.1, 0.0)
    f = C.benthic(base, seabed_z=-3540.0, layer_thickness=60.0, reduction=0.4)
    at_bed = np.hypot(*f(np.array([-3540.0]))[0])
    above = np.hypot(*f(np.array([-3400.0]))[0])
    assert abs(at_bed - 0.04) < 1e-9  # 0.4 * 0.1
    assert abs(above - 0.10) < 1e-9


def test_directional_shear_curves_string_out_of_plane():
    # A current rotating 0->90 deg with depth must bend an ARCA string out of a
    # single vertical plane (both x and y spanned).
    scale = P.calibrate_arca()
    geo = NominalGeometry.load("arca")
    model = P.arca_du_model(geo.units[0], drag_scale=scale)
    z_lo = model.anchor[2]
    z_hi = z_lo + model.rope_length
    p = C.rotating_azimuth(z_lo, z_hi, speed=0.08, azimuth_bottom_deg=0.0, azimuth_top_deg=90.0)
    sh = model.solve(p)
    assert np.ptp(sh.positions[:, 0]) > 1.0
    assert np.ptp(sh.positions[:, 1]) > 1.0
    assert sh.converged


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
