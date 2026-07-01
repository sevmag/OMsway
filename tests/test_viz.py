"""Smoke test for the plotly 3-D visualization utility."""

from __future__ import annotations

import pytest

pytest.importorskip("plotly")

from km3disp import currents as C
from km3disp import viz
from km3disp.displace import displace_arca
from km3disp.geometry import NominalGeometry


def _small_displaced():
    geo = NominalGeometry.load("arca")
    return displace_arca(geo, C.uniform(0.1, 0.0), drag_scale=0.644,
                         solve_kwargs={"n_nodes": 401})


def test_plot_displaced_builds_figure():
    d = _small_displaced()
    fig = viz.plot_displaced(d)
    assert len(fig.data) == 3  # nominal + bent strings + DOM markers
    markers = [t for t in fig.data if t.mode == "markers"][0]
    assert len(markers.x) == d.sensor_ids.size
    assert not fig.layout.sliders  # no size slider


def test_plot_realizations_has_dropdown():
    d = _small_displaced()
    fig = viz.plot_realizations({"a": d, "b": d})
    assert len(fig.data) == 6  # 3 traces x 2 realizations
    assert fig.layout.updatemenus
    assert len(fig.layout.updatemenus[0].buttons) == 2
    assert not fig.layout.sliders  # size slider removed


def test_uniform_sweep_has_speed_slider():
    geo = NominalGeometry.load("arca")
    disp = {s: displace_arca(geo, C.uniform(s / 100.0, 0.0), drag_scale=0.644,
                             solve_kwargs={"n_nodes": 401}) for s in (2, 8, 15)}
    fig = viz.plot_uniform_sweep(disp, default_speed=8)
    assert len(fig.data) == 1 + 2 * 3  # one nominal + (bent, DOM) per speed
    assert len(fig.layout.sliders) == 1  # speed slider only
    assert [s.label for s in fig.layout.sliders[0].steps] == ["2", "8", "15"]


def test_write_html(tmp_path):
    d = _small_displaced()
    out = viz.write_html(viz.plot_displaced(d), tmp_path / "x.html", standalone=False)
    assert out.exists() and out.stat().st_size > 1000


if __name__ == "__main__":
    import pathlib
    import tempfile

    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            if "tmp_path" in fn.__code__.co_varnames:
                with tempfile.TemporaryDirectory() as td:
                    fn(pathlib.Path(td))
            else:
                fn()
            print("PASS", name)
