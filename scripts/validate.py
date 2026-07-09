"""Validate the displacement solver against closed-form and scaling-law limits.

Run in the omsway env:  python scripts/validate.py

All checks (docs/DESIGN.md section 9) run on synthetic strings, so nothing
external is needed:
  * small-angle limit -- a uniform drag load ``q`` on a string under top buoyancy
    ``B`` bends to ``r = q L^2 / (2 B)`` as the current -> 0;
  * arc-length conservation -- the rope is inextensible;
  * quadratic drag -- tip deflection scales as ``v^2`` at small angle.
"""

from __future__ import annotations

import numpy as np

from omsway import Buoy, CylindricalCable, Solver, SphericalOM, String, UniformCurrent

RHO = 1025.0


def _tip(shape, anchor) -> float:
    """Largest horizontal offset of any module from the anchor footprint."""
    d = shape.module_positions[:, :2] - anchor[:2]
    return float(np.hypot(d[:, 0], d[:, 1]).max())


def _arca_like() -> String:
    """A single ~690 m ARCA-like string: 18 near-neutral OMs, a buoy, a cable."""
    anchor = np.array([0.0, 0.0, -3540.0])
    z = -3500.0 + 36.0 * np.arange(18)
    oms = [
        SphericalOM(i, np.array([0.0, 0.0, zi]), radius=0.22, buoyancy=0.0)
        for i, zi in enumerate(z)
    ]
    buoy = Buoy(1350.0, 0.8, 0.25, module_id=99, position=np.array([0.0, 0.0, z[-1] + 38.0]))
    return String(0, anchor, [*oms, buoy], CylindricalCable(0.05, -0.5, c_w=1.2))


def small_angle_limit() -> bool:
    L, B, d, cw = 200.0, 5000.0, 0.05, 1.2
    anchor = np.array([0.0, 0.0, -1000.0])
    string = String(
        0,
        anchor,
        [Buoy(B, c_w=0.0, area=0.0, module_id=0, position=anchor + [0, 0, L])],
        CylindricalCable(d, buoyancy_per_length=0.0, c_w=cw),
    )
    solver = Solver(rho=RHO, n_nodes=8001)
    q = 0.5 * cw * RHO * d  # drag per unit length per unit speed^2
    ok = True
    print(f"  {'u (m/s)':>8} {'numeric':>11} {'analytic':>11} {'ratio':>8}")
    for u in (0.02, 0.05, 0.10):
        r = _tip(solver.solve_string(string, UniformCurrent(u, 0.0)), anchor)
        exact = q * u**2 * L**2 / (2 * B)
        ok &= abs(r / exact - 1.0) < 1e-3
        print(f"  {u:>8.2f} {r:>11.5f} {exact:>11.5f} {r / exact:>8.4f}")
    return ok


def arc_length_conserved() -> bool:
    string = _arca_like()
    sh = Solver().solve_string(string, UniformCurrent(0.15, 30.0))
    path = float(np.linalg.norm(np.diff(sh.positions, axis=0), axis=1).sum())
    print(f"  solved path {path:.3f} m vs rope length {string.rope_length:.3f} m")
    return abs(path - string.rope_length) / string.rope_length < 1e-3


def quadratic_drag() -> bool:
    string, solver = _arca_like(), Solver()
    r1 = _tip(solver.solve_string(string, UniformCurrent(0.02, 0.0)), string.anchor)
    r2 = _tip(solver.solve_string(string, UniformCurrent(0.04, 0.0)), string.anchor)
    print(f"  deflection ratio for 2x speed: {r2 / r1:.3f} (expect ~4)")
    return abs(r2 / r1 - 4.0) < 0.1


def main() -> None:
    ok = True
    for name, check in (
        ("small-angle limit", small_angle_limit),
        ("arc-length conservation", arc_length_conserved),
        ("quadratic (v^2) drag", quadratic_drag),
    ):
        print(name)
        passed = check()
        ok &= passed
        print(f"  -> {'PASS' if passed else 'FAIL'}\n")
    print("ALL PASS" if ok else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
