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
HEADING_COLOR = "#8a4fff"
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

    With ``show_axes`` each module carries two small arrows: one along its
    ``axis`` (the solved tilt) and one along its heading -- the body ``+x``
    reference carried through the tilt and rolled by the torsion. ``axis_scale``
    sets the arrow length as a fraction of the median module spacing (so it reads
    at any detector scale); ``axis_length`` overrides it with metres.
    """
    displaced = geometry.positions()
    nominal = geometry.unperturbed_positions
    offset = np.linalg.norm(displaced - nominal, axis=1)

    modules = [m for s in geometry.strings for m in s]
    axes = np.array([m.axis for m in modules])
    headings = np.array([m.reference_vector() for m in modules])
    tilt_deg = np.degrees([m.tilt for m in modules])
    yaw_deg = np.degrees([m.torsion for m in modules])

    fig = go.Figure()
    for positions, color, name in (
        (nominal, NOMINAL_COLOR, "nominal"),
        (displaced, DISPLACED_COLOR, "displaced"),
    ):
        x, y, z = _polyline(_string_lines(geometry, positions))
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                hoverinfo="skip",
                line=dict(color=color, width=LINE_WIDTH),
                name=name,
            )
        )

    if show_axes:
        length = _auto_axis_length(geometry, axis_scale) if axis_length is None else axis_length
        for vectors, color, name in (
            (axes, AXIS_COLOR, "tilt axis"),
            (headings, HEADING_COLOR, "heading"),
        ):
            gx, gy, gz = _polyline(_arrow_segments(displaced, vectors, length))
            fig.add_trace(
                go.Scatter3d(
                    x=gx,
                    y=gy,
                    z=gz,
                    mode="lines",
                    hoverinfo="skip",
                    line=dict(color=color, width=AXIS_WIDTH),
                    name=name,
                )
            )

    sid, mid = geometry.string_ids(), geometry.module_ids()
    fig.add_trace(
        go.Scatter3d(
            x=displaced[:, 0],
            y=displaced[:, 1],
            z=displaced[:, 2],
            mode="markers",
            name="modules",
            marker=dict(
                size=MODULE_SIZE,
                color=offset,
                colorscale="Viridis",
                cmin=0.0,
                colorbar=dict(
                    title=dict(text="displacement<br>[m]", font=dict(size=11)),
                    len=0.5,
                    thickness=14,
                    y=0.0,
                    yanchor="bottom",
                    tickfont=dict(size=10),
                ),
            ),
            text=[
                f"string {int(s)} · module {int(m)}<br>displacement {o:.1f} m"
                f"<br>tilt {t:.1f}° · yaw {y:.1f}°"
                for s, m, o, t, y in zip(sid, mid, offset, tilt_deg, yaw_deg)
            ],
            hoverinfo="text",
        )
    )

    fig.update_layout(
        title=dict(text=title),
        scene=dict(
            xaxis_title="x [m]", yaxis_title="y [m]", zaxis_title="z [m]", aspectmode="data"
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def _toggle_controls(div_id: str, names: list[str]) -> str:
    """A fixed checkbox panel that shows/hides traces by ``name`` via Plotly.restyle."""
    boxes = "".join(
        f'<label style="display:block;margin:2px 0;">'
        f'<input type="checkbox" data-name="{n}" checked> {n}</label>'
        for n in names
    )
    return f"""
<div id="omsway-toggles" style="position:fixed;top:12px;left:12px;z-index:1000;
     font:13px sans-serif;background:rgba(255,255,255,0.85);color:#222;
     padding:6px 10px;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,0.2);">{boxes}</div>
<script>
(function () {{
  var gd = document.getElementById("{div_id}");
  document.querySelectorAll("#omsway-toggles input").forEach(function (cb) {{
    cb.addEventListener("change", function () {{
      var idx = [];
      gd.data.forEach(function (t, i) {{ if (t.name === cb.dataset.name) idx.push(i); }});
      Plotly.restyle(gd, {{visible: cb.checked ? true : false}}, idx);
    }});
  }});
}})();
</script>"""


def write_html(
    fig: go.Figure,
    path: str | Path,
    *,
    standalone: bool = True,
    toggles: tuple[str, ...] = ("tilt axis", "heading"),
) -> Path:
    """Write an interactive HTML file; ``standalone`` embeds plotly.js for offline use.

    For each trace named in ``toggles`` that the figure contains, a checkbox is
    added to a small panel that shows/hides that trace layer -- so the tilt-axis
    and heading arrows can be switched off and on independently of the legend.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    div_id = "omsway-plot"
    html = fig.to_html(
        include_plotlyjs=(True if standalone else "cdn"), full_html=True, div_id=div_id
    )
    present = [name for name in toggles if any(t.name == name for t in fig.data)]
    if present:
        html = html.replace("</body>", _toggle_controls(div_id, present) + "\n</body>", 1)
    path.write_text(html)
    return path
