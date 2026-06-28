"""Correctness checks for the arc-length force-balance integrator.

Runnable directly (``python tests/test_mechanics.py``) or under pytest.
"""

from __future__ import annotations

import numpy as np

from km3disp.mechanics import DUConstants, DUModel, uniform_current

L = 600.0
ANCHOR = np.array([0.0, 0.0, -3500.0])
DOM_S = np.linspace(40.0, L - 20.0, 18)


def _buoy_dominated(B: float, f_rope: float) -> DUModel:
    """A string whose tension is a constant top buoy with uniform rope drag only.

    This is exactly the regime of the small-angle parabola r(s)=(w/B)(Ls-s^2/2),
    so the numerical integrator must reproduce it as the tilt angle -> 0.
    """
    consts = DUConstants(
        f_dom=0.0, w_dom=0.0, f_rope=f_rope, w_rope=0.0, f_buoy=0.0, w_buoy=B
    )
    return DUModel(anchor=ANCHOR, dom_arclengths=DOM_S, rope_length=L, constants=consts)


def _radial(shape) -> np.ndarray:
    return np.hypot(shape.positions[:, 0] - ANCHOR[0], shape.positions[:, 1] - ANCHOR[1])


def test_small_angle_matches_parabola():
    B, f_rope, U = 3500.0, 1.0, 0.02  # small current -> small tilt
    model = _buoy_dominated(B, f_rope)
    shape = model.solve(uniform_current(U, azimuth_deg=0.0), n_nodes=6001)
    w = f_rope * U**2
    r_parab = (w / B) * (L * shape.s - shape.s**2 / 2)
    r_num = _radial(shape)
    rel = np.abs(r_num[-1] - r_parab[-1]) / r_parab[-1]
    assert shape.max_tilt_deg < 1.0, shape.max_tilt_deg
    assert rel < 1e-3, rel


def test_quadratic_scaling():
    model = _buoy_dominated(3500.0, 1.0)
    r1 = _radial(model.solve(uniform_current(0.02), n_nodes=6001))[-1]
    r2 = _radial(model.solve(uniform_current(0.04), n_nodes=6001))[-1]
    assert abs(r2 / r1 - 4.0) < 1e-2, r2 / r1  # displacement ~ v^2


def test_arclength_conserved():
    # Strong current: large tilt, but height gain must never exceed rope length.
    model = _buoy_dominated(3500.0, 50.0)
    shape = model.solve(uniform_current(0.3), n_nodes=6001)
    height = shape.positions[-1, 2] - ANCHOR[2]
    assert height < L
    assert shape.max_tilt_deg > 5.0  # genuinely large-angle regime
    # path length of the integrated curve is L (unit tangent)
    seg = np.linalg.norm(np.diff(shape.positions, axis=0), axis=1).sum()
    assert abs(seg - L) / L < 1e-3, seg


def test_azimuth_sets_direction():
    model = _buoy_dominated(3500.0, 1.0)
    shape = model.solve(uniform_current(0.05, azimuth_deg=30.0), n_nodes=4001)
    top = shape.positions[-1, :2] - ANCHOR[:2]
    assert abs(np.degrees(np.arctan2(top[1], top[0])) - 30.0) < 1e-2


def test_directional_shear_curves_out_of_plane():
    # Current rotates from 0 deg (bottom) to 90 deg (top): the displaced string
    # must leave a single vertical plane, i.e. span both x and y.
    def sheared(z: np.ndarray) -> np.ndarray:
        frac = np.clip((z - (ANCHOR[2])) / L, 0.0, 1.0)
        ang = np.radians(90.0 * frac)
        speed = 0.1
        return speed * np.column_stack([np.cos(ang), np.sin(ang)])

    consts = DUConstants(
        f_dom=20.0, w_dom=250.0, f_rope=10.0, w_rope=-0.5, f_buoy=100.0, w_buoy=1350.0
    )
    model = DUModel(anchor=ANCHOR, dom_arclengths=DOM_S, rope_length=L, constants=consts)
    shape = model.solve(sheared, n_nodes=4001)
    span_x = np.ptp(shape.positions[:, 0])
    span_y = np.ptp(shape.positions[:, 1])
    assert span_x > 0.5 and span_y > 0.5, (span_x, span_y)
    assert shape.converged


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    # Illustrative run with rough (uncalibrated, Phase-3) ARCA-like constants.
    rough = DUConstants(
        f_dom=62.0, w_dom=250.0, f_rope=9.0, w_rope=-0.5, f_buoy=200.0, w_buoy=1350.0
    )
    m = DUModel(anchor=ANCHOR, dom_arclengths=np.linspace(70, 660, 18), rope_length=690.0, constants=rough)
    for U in (0.05, 0.10, 0.15):
        sh = m.solve(uniform_current(U))
        print(f"  U={U:.2f} m/s  ->  top-DOM deflection {sh.top_dom_deflection:6.2f} m, "
              f"buoy {sh.buoy_deflection:6.2f} m, max tilt {sh.max_tilt_deg:4.1f} deg, "
              f"iters {sh.n_iter}  (rough constants)")
