"""Checks for whole-array displacement and the geometry I/O contract."""

from __future__ import annotations

import numpy as np
import pandas as pd

from km3disp import currents as C
from km3disp.geometry import NominalGeometry, NAMED_GEOMETRIES
from km3disp.displace import displace_arca
from km3disp.io import write_geometry_parquet, write_prometheus_geometry, GEOMETRY_COLUMNS

DRAG_SCALE = 0.644  # fixed to skip recalibration in tests
SOLVE = {"n_nodes": 1001}


def _displaced(tmp_speed=0.07):
    geo = NominalGeometry.load("arca")
    cur = C.uniform(tmp_speed, azimuth_deg=20.0)
    return geo, displace_arca(geo, cur, drag_scale=DRAG_SCALE, solve_kwargs=SOLVE)


def test_array_displacement_basics():
    geo, d = _displaced()
    assert len(d.sensor_ids) == geo.n_sensors == 2070
    assert np.array_equal(d.sensor_ids, np.sort(d.sensor_ids))  # ascending
    # every string displaced downstream; offsets grow with height within a string
    assert d.horizontal_offsets.max() > 1.0
    # anchors fixed: deepest DOM of each string barely moves vs the top DOM
    for u in geo.units[:5]:
        sh = d.shapes[u.string_id]
        off = sh.dom_horizontal_offsets
        assert off[-1] > off[0]  # top moves more than bottom


def test_uniform_direction_is_coherent():
    _, d = _displaced()
    # all strings lean in the same (current) direction -> offset azimuths cluster
    off = d.offsets[:, :2]
    moved = np.hypot(off[:, 0], off[:, 1]) > 0.5
    az = np.degrees(np.arctan2(off[moved, 1], off[moved, 0]))
    assert az.std() < 5.0
    assert abs(np.median(az) - 20.0) < 5.0


def test_geometry_parquet_matches_nominal_schema(tmp_path):
    _, d = _displaced()
    out = write_geometry_parquet(d, tmp_path / "flower_l_displaced.parquet")
    got = pd.read_parquet(out)
    ref = pd.read_parquet(NAMED_GEOMETRIES["arca"])
    assert list(got.columns) == list(ref.columns) == GEOMETRY_COLUMNS
    assert got.index.names == ref.index.names == ["x", "y", "z"]
    assert {c: str(got[c].dtype) for c in got.columns} == {
        c: str(ref[c].dtype) for c in ref.columns
    }
    assert len(got) == len(ref) == 2070
    # the parquet's positions equal the displaced positions exactly (bit-exact)
    assert np.array_equal(got["sensor_pos_x"].to_numpy(), d.positions[:, 0])
    assert np.array_equal(got.index.get_level_values("z").to_numpy(), d.positions[:, 2])
    # same sensor_id set as nominal
    assert set(got["sensor_id"]) == set(ref["sensor_id"])


def test_graphnet_and_prometheus_artifacts_are_bit_identical(tmp_path):
    _, d = _displaced()
    gpath = write_geometry_parquet(d, tmp_path / "g.parquet")
    ppath = write_prometheus_geometry(d, tmp_path / "p.parquet")
    g = pd.read_parquet(gpath)
    p = pd.read_parquet(ppath)
    assert np.array_equal(g["sensor_pos_x"].to_numpy(), p["x"].to_numpy())
    assert np.array_equal(g["sensor_pos_y"].to_numpy(), p["y"].to_numpy())
    assert np.array_equal(g["sensor_pos_z"].to_numpy(), p["z"].to_numpy())


if __name__ == "__main__":
    import tempfile, pathlib

    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            if "tmp_path" in fn.__code__.co_varnames:
                with tempfile.TemporaryDirectory() as td:
                    fn(pathlib.Path(td))
            else:
                fn()
            print("PASS", name)
