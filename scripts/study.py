"""Displacement sweep: bend ARCA under a range of uniform currents and write the
displaced geometries for re-simulation.

Run in the omsway env:  python scripts/study.py [path/to/arca.geo]

Each current speed loads the nominal ARCA geometry, solves the displaced shape,
and writes the displaced .geo to data/displaced/. ARCA hardware constants are
inlined here (see docs/DESIGN.md section 5); drag is uncalibrated
(``drag_scale = 1``) until the benchmark calibration is wired up.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from omsway import Buoy, CylindricalCable, Geometry, Solver, UniformCurrent

ARCA_GEO = Path(
    "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/smagel"
    "/prometheus/resources/geofiles/arca.geo"
)
OUTDIR = Path(__file__).resolve().parent.parent / "data" / "displaced"
SPEEDS_CM = (3, 6, 10, 15)

# ARCA hardware: near-neutral OMs, a syntactic-foam buoy, a bundle of two ropes
# plus a backbone tube; the anchor sits 40 m below the deepest OM.
BUOY = Buoy(1350.0, 0.8, 0.25)  # net buoyancy N, drag coefficient, frontal area m^2
CABLE = CylindricalCable(0.05, -0.5, c_w=1.2)  # bundle width m, net buoyancy N/m
BUOY_GAP = 38.0
Z_FLOOR = -3540.0


def main() -> None:
    geo_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ARCA_GEO
    OUTDIR.mkdir(parents=True, exist_ok=True)
    solver = Solver()
    print(f"{'speed':>7} {'max defl':>10} {'median':>9}  output")
    for cm in SPEEDS_CM:
        geo = Geometry.from_prometheus_geo(
            geo_path, buoy=BUOY, buoy_gap=BUOY_GAP, cable=CABLE, z_floor=Z_FLOOR
        )
        solver.solve(geo, UniformCurrent(cm / 100.0, azimuth_deg=0.0))
        d = np.linalg.norm(geo.displacements(), axis=1)
        out = geo.to_prometheus_geo(OUTDIR / f"arca_{cm:02d}cm.geo")
        print(f"{cm:>5}cm {d.max():>9.1f}m {np.median(d):>8.1f}m  {out.name}")


if __name__ == "__main__":
    main()
