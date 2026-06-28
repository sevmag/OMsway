"""Phase 3 validation of the displacement mechanics and ARCA parameters.

Run ``python -m km3disp.validate`` (or ``python km3disp/validate.py``) for a
report. Two independent legs:

- ANTARES, where the drag/buoyancy constants AND the resulting deflections are
  both published (arXiv:1202.3894), validates the integrator itself.
- ARCA, where KM3NeT's constants are unpublished, is calibrated by a single
  ``drag_scale`` to the published ~100 m @ 0.15 m/s benchmark (arXiv:2007.16090),
  after which the realistic-current regime and v^2 scaling are checked.
"""

from __future__ import annotations

from km3disp.geometry import NominalGeometry
from km3disp.mechanics import uniform_current
from km3disp import parameters as P


def antares_report() -> dict:
    model = P.antares_model()
    out = {}
    for U in (0.05, 0.07, 0.10, 0.20):
        out[U] = model.solve(uniform_current(U)).top_dom_deflection
    return out


def arca_report() -> dict:
    scale = P.calibrate_arca()
    geo = NominalGeometry.load("arca")
    model = P.arca_du_model(geo.units[0], drag_scale=scale)
    regime = {}
    for U in (0.02, 0.03, 0.05, 0.07, 0.10, 0.15):
        sh = model.solve(uniform_current(U))
        regime[U] = (sh.top_dom_deflection, sh.max_tilt_deg)
    # v^2 check at low current (away from large-angle saturation)
    r1 = model.solve(uniform_current(0.02)).top_dom_deflection
    r2 = model.solve(uniform_current(0.04)).top_dom_deflection
    return {"drag_scale": scale, "regime": regime, "v2_ratio": r2 / r1}


def main() -> None:
    print("=" * 64)
    print("ANTARES cross-check (integrator vs published deflections)")
    print("  published: top storey < 2 m at 7 cm/s, ~15 m at 20 cm/s")
    a = antares_report()
    for U, r in a.items():
        print(f"    U = {U*100:4.0f} cm/s   top-storey deflection = {r:6.2f} m")
    assert a[0.07] < 2.0, a[0.07]
    assert 5.0 < a[0.20] < 25.0, a[0.20]
    print("  -> within expectation (mechanics validated)")

    print("=" * 64)
    print("ARCA calibration + regime")
    r = arca_report()
    print(f"  calibrated drag_scale = {r['drag_scale']:.3f}  "
          f"(implied effective cable C_d ~ {1.2 * r['drag_scale']:.2f})")
    print(f"  v^2 scaling check (0.02->0.04 m/s): ratio = {r['v2_ratio']:.2f} (expect ~4)")
    print("  realistic-current regime (nominal ARCA DU, 690 m):")
    for U, (defl, tilt) in r["regime"].items():
        tag = "  <- typical ARCA" if U in (0.02, 0.03) else ""
        print(f"    U = {U:.2f} m/s   top-DOM = {defl:6.2f} m   max tilt = {tilt:4.1f} deg{tag}")
    assert abs(r["v2_ratio"] - 4.0) < 0.3, r["v2_ratio"]
    print("=" * 64)
    print("Phase 3 validation: PASS")


if __name__ == "__main__":
    main()
