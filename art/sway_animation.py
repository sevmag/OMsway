"""DECORATIVE -- not a physics example.

A purely artistic 3-D animation of ARCA's strings swaying. The "current" here is
an invented traveling speed-wave with an exaggerated amplitude chosen to look
good -- not a realistic sea current -- and the output is not validated against
anything. For the actual physics and validation see ``scripts/study.py`` and
``scripts/validate.py``.

Needs plotly + kaleido (kaleido drives a headless Chromium, so give it a few GB
of memory for the 60-frame render). Run:

    python art/sway_animation.py [out.gif] [arca.geo]
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib
import numpy as np
import plotly.graph_objects as go
from PIL import Image

from omsway import Buoy, CylindricalCable, Geometry, Solver
from omsway.currents import CurrentModel

ARCA_GEO = Path(
    "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/smagel"
    "/prometheus/resources/geofiles/arca.geo"
)
BG = "rgba(0,0,0,0)"  # transparent, so it reads on light and dark backgrounds
CAMERA = dict(eye=dict(x=1.15, y=1.02, z=0.42))
# matplotlib "winter" (blue -> green), dropping the bluest 15% (sample t in [0.15, 1])
_w = matplotlib.colormaps["winter"]
WINTER = [[i / 31, "rgb({},{},{})".format(*(int(255 * c) for c in _w(0.15 + 0.85 * i / 31)[:3]))]
          for i in range(32)]


class TravelingWaveCurrent(CurrentModel):
    """A current toward one azimuth whose speed travels that way as a wave.

    ``speed = base + amplitude * sin(k*along - omega*t)``: never reverses (base >
    amplitude), but a crest of faster water sweeps across the array. Invented for
    the animation, not a physical current.
    """

    def __init__(self, base, amplitude, wavelength, period, azimuth_deg=0.0):
        self.base, self.amplitude, self.period = base, amplitude, period
        self.k = 2 * np.pi / wavelength
        self.omega = 2 * np.pi / period
        a = np.radians(azimuth_deg)
        self._dir = np.array([np.cos(a), np.sin(a), 0.0])

    def __call__(self, position, time=0.0):
        p = np.atleast_2d(np.asarray(position, float))
        along = p[:, 0] * self._dir[0] + p[:, 1] * self._dir[1]
        speed = self.base + self.amplitude * np.sin(self.k * along - self.omega * time)
        return speed[:, None] * self._dir


def figure(line, pos, off, cmax, ranges, aspect):
    lx, ly, lz = line
    fig = go.Figure([
        go.Scatter3d(x=lx, y=ly, z=lz, mode="lines", hoverinfo="skip",
                     line=dict(color="#0e7a58", width=1.5)),
        go.Scatter3d(x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], mode="markers", hoverinfo="skip",
                     marker=dict(size=2.4, color=off, colorscale=WINTER, cmin=0.0, cmax=cmax,
                                 showscale=False)),
    ])
    fig.update_layout(
        paper_bgcolor=BG, margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
        scene=dict(bgcolor=BG, aspectmode="manual", aspectratio=aspect, camera=CAMERA,
                   xaxis=dict(visible=False, range=ranges[0]),
                   yaxis=dict(visible=False, range=ranges[1]),
                   zaxis=dict(visible=False, range=ranges[2])),
    )
    return fig


def crop_and_pad(images, margin=10, hpad=48):
    """Crop RGBA frames tight to the detector (by alpha, same box for every frame),
    then add transparent horizontal padding so wrapping text can't touch it."""
    alpha = np.stack([np.asarray(im)[:, :, 3] for im in images])  # (frames, H, W)
    rows = np.where(alpha.any(axis=(0, 2)))[0]
    cols = np.where(alpha.any(axis=(0, 1)))[0]
    y0, y1 = max(0, rows[0] - margin), min(images[0].height, rows[-1] + margin + 1)
    x0, x1 = max(0, cols[0] - margin), min(images[0].width, cols[-1] + margin + 1)
    out = []
    for im in images:
        c = im.crop((x0, y0, x1, y1))
        canvas = Image.new("RGBA", (c.width + 2 * hpad, c.height), (0, 0, 0, 0))
        canvas.paste(c, (hpad, 0))
        out.append(canvas)
    return out


def save_transparent_gif(frames, out, duration=42):
    """Assemble RGBA frames into a looping GIF with a transparent background."""
    palettes = []
    for im in frames:
        p = im.convert("RGB").convert("P", palette=Image.Palette.ADAPTIVE, colors=255,
                                      dither=Image.Dither.NONE)
        p.paste(255, im.split()[3].point(lambda a: 255 if a < 128 else 0))  # 1-bit alpha
        palettes.append(p)
    palettes[0].save(out, save_all=True, append_images=palettes[1:], duration=duration,
                     loop=0, transparency=255, disposal=2)


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("sway.gif")
    geo_path = Path(sys.argv[2]) if len(sys.argv) > 2 else ARCA_GEO

    geo = Geometry.from_prometheus_geo(
        geo_path, buoy=Buoy(1350.0, 0.8, 0.25), buoy_gap=38.0,
        cable=CylindricalCable(0.05, -0.5), z_floor=-3540.0,
    )
    solver = Solver(n_nodes=160)
    wave = TravelingWaveCurrent(base=0.06, amplitude=0.05, wavelength=1300.0, period=1.0)

    nframes, nline = 60, 16
    frames = []
    for t in np.linspace(0.0, wave.period, nframes, endpoint=False):
        shapes = solver.solve(geo, wave, time=t)
        lx, ly, lz = [], [], []
        for s in shapes:
            p = s.positions[np.linspace(0, len(s.s) - 1, nline).astype(int)]
            lx += [*p[:, 0].tolist(), None]
            ly += [*p[:, 1].tolist(), None]
            lz += [*p[:, 2].tolist(), None]
        frames.append(((lx, ly, lz), geo.positions(), np.linalg.norm(geo.displacements(), axis=1)))

    cmax = max(off.max() for _, _, off in frames)
    # Fixed view box centred on the unperturbed centroid, sized to hold every frame,
    # so the anchors map to the same pixels throughout.
    center = geo.unperturbed_positions.mean(axis=0)
    allpos = np.vstack([geo.unperturbed_positions, *[p for _, p, _ in frames]])
    half = 1.05 * np.abs(allpos - center).max(axis=0)
    ranges = [[float(center[i] - half[i]), float(center[i] + half[i])] for i in range(3)]
    spans = [r[1] - r[0] for r in ranges]
    aspect = dict(x=spans[0] / max(spans), y=spans[1] / max(spans), z=spans[2] / max(spans))

    imgs = []
    for k, (line, pos, off) in enumerate(frames):
        png = figure(line, pos, off, cmax, ranges, aspect).to_image(format="png", width=900, height=560)
        imgs.append(Image.open(io.BytesIO(png)).convert("RGBA"))
        if (k + 1) % 15 == 0:
            print(f"rendered {k + 1}/{nframes}")
    save_transparent_gif(crop_and_pad(imgs), out)
    print("wrote", out)


if __name__ == "__main__":
    main()
