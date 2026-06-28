"""Nominal KM3NeT detector geometry.

The nominal geometry is the undisplaced, current-free configuration loaded from a
GraphNeT/NuBench geometry parquet (e.g. ``flower_l.parquet`` for ARCA). In that
state every detection unit is exactly vertical, so all optical modules on a
string share one ``(x, y)`` footprint and differ only in depth ``z``. A detection
unit therefore reduces to an anchor footprint plus a list of module depths, which
is the baseline that the displacement model perturbs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Column schema of the GraphNeT/NuBench per-sensor geometry tables.
XYZ = ["sensor_pos_x", "sensor_pos_y", "sensor_pos_z"]
STRING_COL = "sensor_string_id"
SENSOR_COL = "sensor_id"

# Known geometry tables (vertical-string detectors share this loader).
GEOMETRY_TABLE_DIR = Path(
    "/n/home12/smagel/data/graphnet/data/geometry_tables"
)
NAMED_GEOMETRIES = {
    "arca": GEOMETRY_TABLE_DIR / "nubench" / "flower_l.parquet",
    "orca": GEOMETRY_TABLE_DIR / "nubench" / "flower_s.parquet",
}


@dataclass(frozen=True)
class DetectionUnit:
    """One anchored vertical string of optical modules in the nominal geometry.

    Modules are ordered deepest-first (ascending ``z``); ``sensor_ids`` is aligned
    to that order so displaced positions can be written back to the right rows.
    """

    string_id: int
    anchor_xy: np.ndarray  # (2,) shared (x, y) footprint of the vertical string
    dom_z: np.ndarray  # (n,) nominal module depths, ascending
    sensor_ids: np.ndarray  # (n,) global sensor_id per module

    @property
    def n_dom(self) -> int:
        return len(self.dom_z)

    @property
    def z_bottom(self) -> float:
        return float(self.dom_z[0])

    @property
    def z_top(self) -> float:
        return float(self.dom_z[-1])

    @property
    def instrumented_height(self) -> float:
        """Vertical span from the lowest to the highest module."""
        return self.z_top - self.z_bottom

    @property
    def dom_spacing(self) -> float:
        """Characteristic vertical spacing between modules.

        The median (not mean) is robust to an irregular end gap, so it returns
        the regular inter-DOM spacing rather than being skewed by one interval.
        """
        return float(np.median(np.diff(self.dom_z))) if self.n_dom > 1 else 0.0

    def heights_above(self, datum_z: float) -> np.ndarray:
        """Module heights above a datum (e.g. the seabed anchor depth)."""
        return self.dom_z - datum_z

    def nominal_positions(self) -> np.ndarray:
        """(n, 3) nominal module positions of the vertical string."""
        xy = np.broadcast_to(self.anchor_xy, (self.n_dom, 2))
        return np.column_stack([xy, self.dom_z])


@dataclass
class NominalGeometry:
    """A full detector's nominal geometry as a collection of detection units.

    The source ``table`` is kept verbatim so displaced configurations can be
    written back with an identical schema and row order.
    """

    units: list[DetectionUnit]
    table: pd.DataFrame
    source: Path

    @classmethod
    def from_parquet(
        cls, path: str | Path, *, vertical_tol: float = 1e-3
    ) -> "NominalGeometry":
        path = Path(path)
        df = pd.read_parquet(path)
        missing = [c for c in XYZ + [STRING_COL, SENSOR_COL] if c not in df.columns]
        if missing:
            raise ValueError(f"{path} missing geometry columns: {missing}")

        units: list[DetectionUnit] = []
        for _, g in df.groupby(STRING_COL, sort=True):
            g = g.sort_values("sensor_pos_z")
            x = g["sensor_pos_x"].to_numpy()
            y = g["sensor_pos_y"].to_numpy()
            # The loader expects the nominal current-free geometry; a non-vertical
            # string means an already-displaced table was passed by mistake.
            if np.ptp(x) > vertical_tol or np.ptp(y) > vertical_tol:
                raise ValueError(
                    f"string {int(g[STRING_COL].iloc[0])} is not vertical "
                    f"(x ptp={np.ptp(x):.3g}, "
                    f"y ptp={np.ptp(y):.3g}); from_parquet expects nominal geometry"
                )
            units.append(
                DetectionUnit(
                    string_id=int(g[STRING_COL].iloc[0]),
                    anchor_xy=np.array([x.mean(), y.mean()]),
                    dom_z=g["sensor_pos_z"].to_numpy().copy(),
                    sensor_ids=g[SENSOR_COL].to_numpy().astype(int),
                )
            )
        return cls(units=units, table=df, source=path)

    @classmethod
    def load(cls, name: str = "arca", **kwargs) -> "NominalGeometry":
        if name not in NAMED_GEOMETRIES:
            raise KeyError(f"unknown geometry '{name}'; known: {list(NAMED_GEOMETRIES)}")
        return cls.from_parquet(NAMED_GEOMETRIES[name], **kwargs)

    def __len__(self) -> int:
        return len(self.units)

    def __iter__(self):
        return iter(self.units)

    def __getitem__(self, string_id: int) -> DetectionUnit:
        for u in self.units:
            if u.string_id == string_id:
                return u
        raise KeyError(string_id)

    @property
    def n_strings(self) -> int:
        return len(self.units)

    @property
    def n_dom_per_string(self) -> int:
        counts = {u.n_dom for u in self.units}
        return counts.pop() if len(counts) == 1 else -1

    @property
    def n_sensors(self) -> int:
        return sum(u.n_dom for u in self.units)

    @property
    def anchors_xy(self) -> np.ndarray:
        """(n_strings, 2) anchor footprints."""
        return np.array([u.anchor_xy for u in self.units])

    def nearest_neighbour_spacing(self) -> np.ndarray:
        """Per-string distance to the closest other string footprint."""
        xy = self.anchors_xy
        d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        return np.sqrt(d2.min(1))

    def summary(self) -> dict:
        z = self.table["sensor_pos_z"].to_numpy()
        nn = self.nearest_neighbour_spacing()
        return {
            "source": str(self.source),
            "n_strings": self.n_strings,
            "n_dom_per_string": self.n_dom_per_string,
            "n_sensors": self.n_sensors,
            "dom_spacing_m": float(np.median([u.dom_spacing for u in self.units])),
            "instrumented_height_m": float(
                np.median([u.instrumented_height for u in self.units])
            ),
            "z_bottom_m": float(z.min()),
            "z_top_m": float(z.max()),
            "footprint_radius_m": float(np.hypot(*self.anchors_xy.T).max()),
            "string_spacing_m": float(np.median(nn)),
        }


if __name__ == "__main__":
    geo = NominalGeometry.load("arca")
    s = geo.summary()
    print("Nominal ARCA geometry")
    for k, v in s.items():
        print(f"  {k:24s} {v}")
    u = geo.units[0]
    print(f"\nstring {u.string_id}: anchor_xy={u.anchor_xy.round(1)}, "
          f"{u.n_dom} DOMs, z {u.z_bottom:.0f}..{u.z_top:.0f} m, "
          f"spacing {u.dom_spacing:.0f} m")
