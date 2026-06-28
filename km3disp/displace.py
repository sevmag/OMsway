"""Apply a current field to a whole detector and collect displaced positions.

The horizontal current decorrelates over scales much larger than the ~1 km array
footprint, so every detection unit sees the same field; anchors stay fixed and
the strings bend coherently. Each string is solved independently with the
arc-length model and the per-DOM displaced positions are gathered, ordered by
``sensor_id`` to match the nominal geometry-table convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import parameters as P
from .geometry import NominalGeometry
from .mechanics import CurrentField, DUModel, DUShape

ModelBuilder = Callable[[object], DUModel]


@dataclass
class DisplacedArray:
    """Displaced positions of every sensor under one current field."""

    geometry: NominalGeometry
    current: CurrentField
    shapes: dict  # string_id -> DUShape
    sensor_ids: np.ndarray  # (N,) ascending
    string_ids: np.ndarray  # (N,) aligned with sensor_ids
    positions: np.ndarray  # (N, 3) displaced, aligned with sensor_ids
    nominal_positions: np.ndarray  # (N, 3) aligned with sensor_ids

    @property
    def offsets(self) -> np.ndarray:
        return self.positions - self.nominal_positions

    @property
    def horizontal_offsets(self) -> np.ndarray:
        d = self.offsets
        return np.hypot(d[:, 0], d[:, 1])

    def summary(self) -> dict:
        h = self.horizontal_offsets
        return {
            "n_sensors": len(self.sensor_ids),
            "max_horizontal_offset_m": float(h.max()),
            "median_horizontal_offset_m": float(np.median(h)),
            "max_vertical_drop_m": float((-self.offsets[:, 2]).max()),
        }


def displace_array(
    geometry: NominalGeometry,
    current: CurrentField,
    model_builder: ModelBuilder,
    *,
    solve_kwargs: dict | None = None,
) -> DisplacedArray:
    shapes: dict = {}
    sid, strid, pos, nom = [], [], [], []
    for unit in geometry:
        shape: DUShape = model_builder(unit).solve(current, **(solve_kwargs or {}))
        shapes[unit.string_id] = shape
        sid.append(unit.sensor_ids)
        strid.append(np.full(unit.n_dom, unit.string_id))
        pos.append(shape.dom_positions)
        nom.append(unit.nominal_positions())

    sensor_ids = np.concatenate(sid)
    order = np.argsort(sensor_ids, kind="stable")
    return DisplacedArray(
        geometry=geometry,
        current=current,
        shapes=shapes,
        sensor_ids=sensor_ids[order],
        string_ids=np.concatenate(strid)[order],
        positions=np.concatenate(pos)[order],
        nominal_positions=np.concatenate(nom)[order],
    )


def arca_model_builder(drag_scale: float | None = None, **constant_kwargs) -> ModelBuilder:
    """A model builder for ARCA strings; calibrates ``drag_scale`` if not given."""
    if drag_scale is None:
        drag_scale = P.calibrate_arca(**constant_kwargs)
    return lambda unit: P.arca_du_model(unit, drag_scale=drag_scale, **constant_kwargs)


def displace_arca(
    geometry: NominalGeometry,
    current: CurrentField,
    *,
    drag_scale: float | None = None,
    solve_kwargs: dict | None = None,
    **constant_kwargs,
) -> DisplacedArray:
    """Displace a nominal ARCA geometry under a current field."""
    builder = arca_model_builder(drag_scale=drag_scale, **constant_kwargs)
    return displace_array(geometry, current, builder, solve_kwargs=solve_kwargs)
