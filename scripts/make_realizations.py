"""Generate several water-sway realizations of ARCA and an interactive 3-D viewer.

Reproducible entry point: ``python scripts/make_realizations.py``. Displaces the
nominal ARCA geometry under a handful of distinct current settings (calm, strong,
depth-sheared speed, depth-rotating direction, near-bottom suppressed) and writes
a single self-contained plotly HTML with a dropdown to switch between them, to
``data/realizations/arca_sway.html``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from km3disp import currents as C
from km3disp import parameters as P
from km3disp import viz
from km3disp.displace import displace_arca
from km3disp.geometry import NominalGeometry

SOLVE = {"n_nodes": 1001}


def main() -> None:
    geo = NominalGeometry.load("arca")
    drag_scale = P.calibrate_arca()
    # depth span of a string (anchor -> buoy), for depth-resolved profiles
    model0 = P.arca_du_model(geo.units[0], drag_scale=drag_scale)
    z_lo = float(model0.anchor[2])
    z_hi = z_lo + model0.rope_length

    settings = {
        "calm 3 cm/s": C.uniform(0.03, azimuth_deg=0.0),
        "moderate 8 cm/s @45deg": C.uniform(0.08, azimuth_deg=45.0),
        "strong 15 cm/s @90deg": C.uniform(0.15, azimuth_deg=90.0),
        "sheared (weak bottom, strong top)": C.sheared_speed(
            z_lo, z_hi, speed_bottom=0.03, speed_top=0.13, azimuth_deg=0.0),
        "directional shear (rotates 0->120deg)": C.rotating_azimuth(
            z_lo, z_hi, speed=0.10, azimuth_bottom_deg=0.0, azimuth_top_deg=120.0),
        "benthic storm (15 cm/s, suppressed near bed)": C.benthic(
            C.uniform(0.15, azimuth_deg=60.0), seabed_z=z_lo, layer_thickness=80.0,
            reduction=0.4),
    }

    realizations = {}
    print(f"{'realization':46} {'max offset [m]':>14} {'top tilt [deg]':>15}")
    for name, current in settings.items():
        disp = displace_arca(geo, current, drag_scale=drag_scale, solve_kwargs=SOLVE)
        realizations[name] = disp
        max_tilt = max(s.max_tilt_deg for s in disp.shapes.values())
        print(f"{name:46} {disp.summary()['max_horizontal_offset_m']:14.1f} {max_tilt:15.1f}")

    fig = viz.plot_realizations(realizations)
    out = viz.write_html(fig, ROOT / "data" / "realizations" / "arca_sway.html")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
