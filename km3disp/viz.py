"""Interactive 3-D visualization of displaced detector configurations (plotly).

Three viewers:
- ``plot_displaced`` -- a single configuration.
- ``plot_realizations`` -- several named configurations with a dropdown.
- ``plot_uniform_sweep`` -- a uniform current along +x with a current-speed slider
  that switches between pre-computed displacements.

Each renders detection units as their bent arc-length shape (the displaced "real"
strings) with the optical modules coloured by horizontal offset, and the nominal
vertical strings drawn for reference. Nominal and displaced strings use the same
line width so the offset between them is what stands out.

plotly is an optional dependency (extras ``plot``); importing this module
requires it, but the rest of ``km3disp`` does not.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from .displace import DisplacedArray

_PALETTE = ["cornflowerblue", "firebrick", "seagreen", "darkorange", "purple", "teal"]
_BENT_COLOR = "cornflowerblue"

DOM_SIZE = 3.0
NOMINAL_COLOR = "grey"
LINE_WIDTH = 3.0  # both nominal and displaced strings


def _line_coords(segments):
    """Flatten a list of (n,3) polylines into x/y/z with None breaks between them."""
    xs, ys, zs = [], [], []
    for seg in segments:
        xs.extend(seg[:, 0].tolist() + [None])
        ys.extend(seg[:, 1].tolist() + [None])
        zs.extend(seg[:, 2].tolist() + [None])
    return xs, ys, zs


def _bent_segments(displaced, n_line):
    out = []
    for unit in displaced.geometry:
        shape = displaced.shapes[unit.string_id]
        idx = np.linspace(0, len(shape.s) - 1, n_line).astype(int)
        out.append(shape.positions[idx])
    return out


def _nominal_segments(displaced):
    return [np.vstack([displaced.shapes[u.string_id].anchor, u.nominal_positions()])
            for u in displaced.geometry]


def _nominal_trace(displaced, visible=True):
    xs, ys, zs = _line_coords(_nominal_segments(displaced))
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", visible=visible, hoverinfo="skip",
                        line=dict(color=NOMINAL_COLOR, width=LINE_WIDTH), name="nominal")


def _bent_trace(displaced, n_line, color, visible=True):
    xs, ys, zs = _line_coords(_bent_segments(displaced, n_line))
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", visible=visible, hoverinfo="skip",
                        line=dict(color=color, width=LINE_WIDTH), name="displaced strings")


def _dom_trace(displaced, visible=True):
    off = displaced.horizontal_offsets
    return go.Scatter3d(
        x=displaced.positions[:, 0], y=displaced.positions[:, 1], z=displaced.positions[:, 2],
        mode="markers", visible=visible, name="DOMs",
        marker=dict(size=DOM_SIZE, color=off, colorscale="Viridis", cmin=0.0,
                    colorbar=dict(title="horiz.<br>offset [m]")),
        text=[f"string {int(s)} · DOM {int(i)}<br>offset {o:.1f} m"
              for s, i, o in zip(displaced.string_ids, displaced.sensor_ids, off)],
        hoverinfo="text")


def _add_realization(fig, displaced, *, n_line, visible, color, show_nominal=True):
    """Add one configuration's traces; return the list of their indices."""
    start = len(fig.data)
    if show_nominal:
        fig.add_trace(_nominal_trace(displaced, visible))
    fig.add_trace(_bent_trace(displaced, n_line, color, visible))
    fig.add_trace(_dom_trace(displaced, visible))
    return list(range(start, len(fig.data)))


def _scene_layout(title):
    return dict(
        title=dict(text=title),
        scene=dict(xaxis_title="x [m]", yaxis_title="y [m]", zaxis_title="z [m]",
                   aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
    )


def plot_displaced(displaced: DisplacedArray, *, title="Displaced ARCA",
                   n_line: int = 40, show_nominal: bool = True) -> go.Figure:
    """A 3-D figure of a single displaced configuration."""
    fig = go.Figure()
    _add_realization(fig, displaced, n_line=n_line, visible=True,
                     color=_PALETTE[0], show_nominal=show_nominal)
    fig.update_layout(**_scene_layout(title))
    return fig


def plot_realizations(realizations: dict, *, n_line: int = 40,
                      show_nominal: bool = True) -> go.Figure:
    """Overlay several named configurations with a dropdown to switch between them."""
    fig = go.Figure()
    names = list(realizations)
    idx_map = {}
    for k, name in enumerate(names):
        idx_map[name] = _add_realization(
            fig, realizations[name], n_line=n_line, visible=(k == 0),
            color=_PALETTE[k % len(_PALETTE)], show_nominal=show_nominal,
        )
    n_traces = len(fig.data)
    buttons = []
    for name in names:
        vis = [False] * n_traces
        for j in idx_map[name]:
            vis[j] = True
        buttons.append(dict(label=name, method="update",
                            args=[{"visible": vis},
                                  {"title.text": f"ARCA water sway — {name}"}]))
    fig.update_layout(**_scene_layout(f"ARCA water sway — {names[0]}"))
    fig.update_layout(updatemenus=[dict(buttons=buttons, direction="down",
                                        x=0.0, y=1.0, xanchor="left", yanchor="top",
                                        showactive=True)])
    return fig


def plot_uniform_sweep(displaced_by_speed: dict, *, n_line: int = 40,
                       default_speed=None) -> go.Figure:
    """Uniform-current viewer with a current-speed slider.

    ``displaced_by_speed`` maps a speed label (e.g. cm/s) to a ``DisplacedArray``.
    The nominal strings are drawn once (speed-independent); the speed slider
    toggles between the per-speed displaced strings + DOMs.
    """
    speeds = sorted(displaced_by_speed)
    if default_speed is None:
        default_speed = speeds[len(speeds) // 2]
    fig = go.Figure()
    fig.add_trace(_nominal_trace(displaced_by_speed[speeds[0]]))
    speed_traces = {}
    for speed in speeds:
        disp = displaced_by_speed[speed]
        visible = speed == default_speed
        fig.add_trace(_bent_trace(disp, n_line, _BENT_COLOR, visible))
        bent_i = len(fig.data) - 1
        fig.add_trace(_dom_trace(disp, visible))
        dom_i = len(fig.data) - 1
        speed_traces[speed] = (bent_i, dom_i)

    n_traces = len(fig.data)
    steps = []
    for speed in speeds:
        vis = [False] * n_traces
        vis[0] = True  # nominal always shown
        b, d = speed_traces[speed]
        vis[b] = vis[d] = True
        steps.append(dict(method="update", label=str(speed),
                          args=[{"visible": vis},
                                {"title.text": f"Uniform current along +x — {speed} cm/s"}]))
    speed_slider = dict(active=speeds.index(default_speed), steps=steps,
                        x=0.0, y=0.0, len=0.9, pad={"t": 30, "b": 10},
                        currentvalue={"prefix": "current speed: ", "suffix": " cm/s"})
    fig.update_layout(**_scene_layout(f"Uniform current along +x — {default_speed} cm/s"))
    fig.update_layout(sliders=[speed_slider])
    return fig


def write_html(fig: go.Figure, path: str | Path, *, standalone: bool = True) -> Path:
    """Write an interactive HTML file. ``standalone`` embeds plotly.js (offline-viewable)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs=(True if standalone else "cdn"))
    return path
