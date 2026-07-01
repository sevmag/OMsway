"""Generate a uniform-current viewer with a current-speed slider.

Reproducible entry point: ``python scripts/make_uniform_sweep.py``. Displaces
ARCA under a uniform current pointing along +x (azimuth 0) for speeds 2-15 cm/s
and writes a single interactive HTML with a speed slider (plus the DOM-size /
nominal-width slider) to ``data/realizations/arca_uniform_speed.html``.
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

SPEEDS_CM = list(range(2, 16))  # 2..15 cm/s
DEFAULT_CM = 8
SOLVE = {"n_nodes": 801}


def main() -> None:
    geo = NominalGeometry.load("arca")
    drag_scale = P.calibrate_arca()
    displaced = {}
    print(f"{'speed [cm/s]':>12} {'max offset [m]':>15}")
    for speed_cm in SPEEDS_CM:
        disp = displace_arca(geo, C.uniform(speed_cm / 100.0, azimuth_deg=0.0),
                             drag_scale=drag_scale, solve_kwargs=SOLVE)
        displaced[speed_cm] = disp
        print(f"{speed_cm:12d} {disp.summary()['max_horizontal_offset_m']:15.1f}")

    fig = viz.plot_uniform_sweep(displaced, default_speed=DEFAULT_CM)
    out = viz.write_html(fig, ROOT / "data" / "realizations" / "arca_uniform_speed.html")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
