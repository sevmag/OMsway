"""Interactive 3-D view of a displaced detector (plotly).

Each string is drawn as a line from its anchor up through its modules, once for
the nominal baseline (grey) and once for the displaced positions (coloured); the
modules are marked and coloured by how far they moved, and a short glyph from
each module shows its axis (the solved tilt). Pass a geometry that has been
solved (``Solver.solve``), so it carries both the displaced positions and the
nominal baseline.

plotly is an optional dependency; importing this module needs it, the rest of
``omsway`` does not.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from .geometry import Geometry

NOMINAL_COLOR = "grey"
DISPLACED_COLOR = "cornflowerblue"
AXIS_COLOR = "#e4572e"
LINE_WIDTH = 3.0
AXIS_WIDTH = 4.0
MODULE_SIZE = 3.0


def _polyline(segments):
    """Flatten ``(n, 3)`` polylines into x/y/z lists with ``None`` breaks between."""
    xs, ys, zs = [], [], []
    for seg in segments:
        xs += [*seg[:, 0].tolist(), None]
        ys += [*seg[:, 1].tolist(), None]
        zs += [*seg[:, 2].tolist(), None]
    return xs, ys, zs


def _string_lines(geometry: Geometry, positions: np.ndarray):
    """One polyline per string: its anchor followed by its module positions."""
    lines, i = [], 0
    for s in geometry.strings:
        lines.append(np.vstack([s.anchor, positions[i : i + s.n_modules]]))
        i += s.n_modules
    return lines


def _auto_axis_length(geometry: Geometry, factor: float = 0.3) -> float:
    """Arrow length as ``factor`` times the median module spacing, so it reads at any scale."""
    gaps, i, pos = [], 0, geometry.positions()
    for s in geometry.strings:
        block = pos[i : i + s.n_modules]
        i += s.n_modules
        if len(block) > 1:
            gaps.append(np.linalg.norm(np.diff(block, axis=0), axis=1))
    return factor * float(np.median(np.concatenate(gaps))) if gaps else factor * 25.0


def _arrow_segments(bases, axes, length, head=0.35, spread=0.18):
    """Line segments for a small 3-D arrow per (base, axis): a shaft to the tip
    plus a four-barb head. Everything is in data units, so the arrow keeps a
    predictable size at any scene scale (unlike a plotly Cone, whose ``sizeref``
    tracks the scene extent)."""
    ref = np.tile([0.0, 0.0, 1.0], (len(axes), 1))
    ref[np.abs(axes[:, 2]) > 0.99] = [1.0, 0.0, 0.0]  # avoid a degenerate cross
    e1 = np.cross(axes, ref)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(axes, e1)
    tips = bases + length * axes
    back = tips - head * length * axes
    segs = []
    for i in range(len(axes)):
        segs.append(np.vstack([bases[i], tips[i]]))  # shaft
        for e in (e1[i], e2[i]):
            segs.append(np.vstack([tips[i], back[i] + spread * length * e]))
            segs.append(np.vstack([tips[i], back[i] - spread * length * e]))
    return segs


def plot(
    geometry: Geometry,
    *,
    title: str = "Displaced detector",
    show_axes: bool = True,
    axis_length: float | None = None,
    axis_scale: float = 0.3,
) -> go.Figure:
    """3-D figure of the displaced geometry against its nominal baseline.

    With ``show_axes`` each module carries a small arrow along its ``axis`` (the
    solved tilt). ``axis_scale`` sets the arrow length as a fraction of the median
    module spacing (so it reads at any detector scale); ``axis_length`` overrides
    it with an absolute length in metres.
    """
    displaced = geometry.positions()
    nominal = geometry.unperturbed_positions
    offset = np.linalg.norm(displaced - nominal, axis=1)

    modules = [m for s in geometry.strings for m in s]
    axes = np.array([m.axis for m in modules])
    tilt_deg = np.degrees([m.tilt for m in modules])
    yaw_deg = np.degrees([m.torsion for m in modules])

    fig = go.Figure()
    for positions, color, name in (
        (nominal, NOMINAL_COLOR, "nominal"),
        (displaced, DISPLACED_COLOR, "displaced"),
    ):
        x, y, z = _polyline(_string_lines(geometry, positions))
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", hoverinfo="skip",
                                   line=dict(color=color, width=LINE_WIDTH), name=name))

    if show_axes:
        length = _auto_axis_length(geometry, axis_scale) if axis_length is None else axis_length
        ax_x, ax_y, ax_z = _polyline(_arrow_segments(displaced, axes, length))
        fig.add_trace(go.Scatter3d(x=ax_x, y=ax_y, z=ax_z, mode="lines", hoverinfo="skip",
                                   line=dict(color=AXIS_COLOR, width=AXIS_WIDTH), name="tilt axis"))

    sid, mid = geometry.string_ids(), geometry.module_ids()
    fig.add_trace(go.Scatter3d(
        x=displaced[:, 0], y=displaced[:, 1], z=displaced[:, 2], mode="markers", name="modules",
        marker=dict(size=MODULE_SIZE, color=offset, colorscale="Viridis", cmin=0.0,
                    colorbar=dict(title="displacement<br>[m]")),
        text=[f"string {int(s)} · module {int(m)}<br>displacement {o:.1f} m"
              f"<br>tilt {t:.1f}° · yaw {y:.1f}°"
              for s, m, o, t, y in zip(sid, mid, offset, tilt_deg, yaw_deg)],
        hoverinfo="text"))

    fig.update_layout(
        title=dict(text=title),
        scene=dict(xaxis_title="x [m]", yaxis_title="y [m]", zaxis_title="z [m]",
                   aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def write_html(fig: go.Figure, path: str | Path, *, standalone: bool = True) -> Path:
    """Write an interactive HTML file; ``standalone`` embeds plotly.js for offline use."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs=(True if standalone else "cdn"))
    return path
