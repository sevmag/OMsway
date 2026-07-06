"""Interactive 3-D view of a displaced detector (plotly).

Each string is drawn as a line from its anchor up through its modules, once for
the nominal baseline (grey) and once for the displaced positions (coloured); the
modules are marked and coloured by how far they moved. Pass a geometry that has
been solved (``Solver.solve``), so it carries both the displaced positions and
the nominal baseline.

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
LINE_WIDTH = 3.0
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


def plot(geometry: Geometry, *, title: str = "Displaced detector") -> go.Figure:
    """3-D figure of the displaced geometry against its nominal baseline."""
    displaced = geometry.positions()
    nominal = geometry.unperturbed_positions
    offset = np.linalg.norm(displaced - nominal, axis=1)

    fig = go.Figure()
    for positions, color, name in (
        (nominal, NOMINAL_COLOR, "nominal"),
        (displaced, DISPLACED_COLOR, "displaced"),
    ):
        x, y, z = _polyline(_string_lines(geometry, positions))
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", hoverinfo="skip",
                                   line=dict(color=color, width=LINE_WIDTH), name=name))

    sid, mid = geometry.string_ids(), geometry.module_ids()
    fig.add_trace(go.Scatter3d(
        x=displaced[:, 0], y=displaced[:, 1], z=displaced[:, 2], mode="markers", name="modules",
        marker=dict(size=MODULE_SIZE, color=offset, colorscale="Viridis", cmin=0.0,
                    colorbar=dict(title="displacement<br>[m]")),
        text=[f"string {int(s)} · module {int(m)}<br>{o:.1f} m"
              for s, m, o in zip(sid, mid, offset)],
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
