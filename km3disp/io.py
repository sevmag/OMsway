"""Write displaced geometry in the formats GraphNeT and Prometheus consume.

Both artifacts are built from the same displaced-position array, so the GraphNeT
geometry-table index and the Prometheus per-sensor positions are numerically
identical -- required because GraphNeT's DataRepresentation matches hit xyz
against the geometry-table index by exact float value (see docs/DESIGN.md 7.2).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .displace import DisplacedArray

# Column schema and dtypes of the GraphNeT/NuBench geometry tables (flower_l.parquet).
GEOMETRY_COLUMNS = [
    "sensor_pos_x",
    "sensor_pos_y",
    "sensor_pos_z",
    "sensor_string_id",
    "sensor_id",
    "t",
]


def build_geometry_frame(
    sensor_ids: np.ndarray, string_ids: np.ndarray, positions: np.ndarray
) -> pd.DataFrame:
    """A geometry DataFrame matching ``flower_l.parquet`` (MultiIndex + columns)."""
    df = pd.DataFrame(
        {
            "sensor_pos_x": positions[:, 0],
            "sensor_pos_y": positions[:, 1],
            "sensor_pos_z": positions[:, 2],
            "sensor_string_id": string_ids.astype(np.float64),
            "sensor_id": sensor_ids.astype(np.int64),
            "t": np.zeros(len(sensor_ids), dtype=np.int64),
        }
    )
    df.index = pd.MultiIndex.from_arrays(
        [positions[:, 0], positions[:, 1], positions[:, 2]], names=["x", "y", "z"]
    )
    return df


def write_geometry_parquet(displaced: DisplacedArray, path: str | Path) -> Path:
    """Write the displaced GraphNeT geometry table (mode (a) reconstruction)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = build_geometry_frame(
        displaced.sensor_ids, displaced.string_ids, displaced.positions
    )
    frame.to_parquet(path)
    return path


def write_prometheus_geometry(displaced: DisplacedArray, path: str | Path) -> Path:
    """Write the Prometheus-side per-sensor positions (simulation input).

    A plain table of (string_id, sensor_id, x, y, z) built from the SAME position
    array as the GraphNeT parquet, so the two are bit-identical. The exact handoff
    to the Prometheus geometry object is a Phase-6 integration detail; this file is
    the single source of truth for those coordinates.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "string_id": displaced.string_ids.astype(np.int64),
            "sensor_id": displaced.sensor_ids.astype(np.int64),
            "x": displaced.positions[:, 0],
            "y": displaced.positions[:, 1],
            "z": displaced.positions[:, 2],
        }
    ).to_parquet(path, index=False)
    return path
