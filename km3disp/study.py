"""Phase 6: NuBench-style study of reconstruction degradation under displacement.

Two layers:

1. The integration harness for the full pipeline -- a displacement sweep that
   writes displaced geometries, a GraphNeT ``Detector`` hook pointing at a
   displaced table, and the mode-(b) "calibration-lag" data transform. Running
   the full pipeline needs Prometheus (photon simulation) and GPU training of a
   GraphNeT model, neither executed here.

2. A self-contained physics proxy that runs end-to-end with numpy now: synthetic
   muon tracks, Cherenkov first-photon hit times computed on the TRUE (displaced)
   geometry, and a line-fit direction reconstruction. Reconstructing with the
   displaced positions (mode a, intrinsic) vs the nominal positions (mode b,
   calibration-lag) isolates the two effects and yields the degradation-vs-
   displacement curve.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .currents import uniform
from .displace import DisplacedArray, arca_model_builder, displace_array
from .geometry import NominalGeometry
from .io import write_geometry_parquet

C_LIGHT = 0.299792458  # m/ns, vacuum
N_WATER = 1.364  # KM3NeT seawater group index (relevant band)
CHERENKOV_FACTOR = np.sqrt(N_WATER**2 - 1.0)


# --- Full-pipeline harness ------------------------------------------------

def displacement_sweep(
    geometry: NominalGeometry,
    speeds,
    *,
    azimuth_deg: float = 0.0,
    drag_scale: float = 0.644,
    outdir: str | Path | None = None,
    solve_nodes: int = 1501,
) -> dict:
    """Displace ``geometry`` at each current speed; optionally write parquets."""
    builder = arca_model_builder(drag_scale=drag_scale)
    out: dict = {}
    for speed in speeds:
        disp = displace_array(
            geometry, uniform(speed, azimuth_deg), builder,
            solve_kwargs={"n_nodes": solve_nodes},
        )
        if outdir is not None:
            write_geometry_parquet(disp, Path(outdir) / f"arca_u{speed:.3f}.parquet")
        out[speed] = disp
    return out


def make_displaced_detector(geometry_table_path: str | Path):
    """A GraphNeT ``FlowerL`` (ARCA) detector pointed at a displaced geometry table.

    This is the one-line reconstruction-side hook for mode (a). ``graphnet`` is a
    heavy optional dependency only needed for the full ML pipeline (not the proxy
    below), so it is imported here rather than at module load.
    """
    from graphnet.models.detector.nubench import FlowerL

    class FlowerLDisplaced(FlowerL):
        pass

    FlowerLDisplaced.geometry_table_path = str(geometry_table_path)
    return FlowerLDisplaced


def relabel_hits_to_nominal(
    hit_positions: np.ndarray, displaced: DisplacedArray
) -> np.ndarray:
    """Mode (b): map each hit's displaced position to its DOM's nominal position.

    The realistic calibration-lag failure: the detector is physically displaced
    (so photon arrival times reflect the true geometry) but reconstruction is fed
    nominal positions. Hits are matched to the nearest displaced DOM and replaced
    by that DOM's nominal coordinate, leaving the (displaced-geometry) times.
    """
    disp = displaced.positions
    nom = displaced.nominal_positions
    # nearest displaced DOM per hit (exact match when hits sit on DOMs)
    d2 = ((hit_positions[:, None, :] - disp[None, :, :]) ** 2).sum(-1)
    idx = d2.argmin(1)
    return nom[idx]


# --- Physics proxy --------------------------------------------------------

def cherenkov_hit_times(
    track_point: np.ndarray, track_dir: np.ndarray, dom_pos: np.ndarray
) -> np.ndarray:
    """Direct Cherenkov first-photon arrival time at each DOM (ns), t0 = 0.

    ``t = (1/c)[ d.(p - r0) + rho * sqrt(n^2 - 1) ]`` for a muon at ~c along
    ``d`` through ``r0``, with ``rho`` the perpendicular distance to the track.
    """
    rel = dom_pos - track_point
    along = rel @ track_dir
    perp = np.linalg.norm(rel - along[:, None] * track_dir, axis=1)
    return (along + perp * CHERENKOV_FACTOR) / C_LIGHT


def _perp_distance(track_point, track_dir, dom_pos) -> np.ndarray:
    rel = dom_pos - track_point
    along = rel @ track_dir
    return np.linalg.norm(rel - along[:, None] * track_dir, axis=1)


def line_fit_direction(pos: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Least-squares line fit ``pos ~ r0 + v t``; returns the unit velocity.

    The velocity points in the direction of increasing time, i.e. the track's
    direction of travel. Sensitive to the assumed sensor positions, which is what
    makes it a probe of displacement error.
    """
    tc = t - t.mean()
    pc = pos - pos.mean(0)
    denom = (tc * tc).sum()
    if denom <= 0:
        return np.array([0.0, 0.0, 1.0])
    v = (pc * tc[:, None]).sum(0) / denom
    n = np.linalg.norm(v)
    return v / n if n > 0 else np.array([0.0, 0.0, 1.0])


def opening_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))


def _sample_tracks(rng, n_events, geometry):
    tab = geometry.table
    cx = float(tab["sensor_pos_x"].mean())
    cy = float(tab["sensor_pos_y"].mean())
    zlo = float(tab["sensor_pos_z"].min())
    zhi = float(tab["sensor_pos_z"].max())
    radius = float(np.hypot(tab["sensor_pos_x"] - cx, tab["sensor_pos_y"] - cy).max())
    costh = rng.uniform(-1.0, 1.0, n_events)
    sinth = np.sqrt(1.0 - costh**2)
    phi = rng.uniform(0.0, 2 * np.pi, n_events)
    dirs = np.column_stack([sinth * np.cos(phi), sinth * np.sin(phi), costh])
    pts = np.column_stack([
        cx + rng.uniform(-0.6 * radius, 0.6 * radius, n_events),
        cy + rng.uniform(-0.6 * radius, 0.6 * radius, n_events),
        rng.uniform(zlo, zhi, n_events),
    ])
    return dirs, pts


def proxy_study(
    geometry: NominalGeometry,
    speeds,
    *,
    n_events: int = 500,
    seed: int = 0,
    r_hit: float = 150.0,
    min_hits: int = 12,
    sigma_t_ns: float = 2.0,
    drag_scale: float = 0.644,
    solve_nodes: int = 801,
) -> dict:
    """Median reconstruction error vs current speed for modes (a) and (b).

    The same set of tracks (and the same per-hit time noise) is reused across all
    speeds so differences are purely the displacement effect.
    """
    rng = np.random.default_rng(seed)
    builder = arca_model_builder(drag_scale=drag_scale)
    dirs, points = _sample_tracks(rng, n_events, geometry)
    noise = rng.normal(0.0, sigma_t_ns, (n_events, geometry.n_sensors))

    out: dict = {}
    for speed in speeds:
        disp = displace_array(
            geometry, uniform(speed, 0.0), builder, solve_kwargs={"n_nodes": solve_nodes}
        )
        true_pos = disp.positions
        nom_pos = disp.nominal_positions
        ang_a, ang_b = [], []
        for k in range(n_events):
            d, p0 = dirs[k], points[k]
            sel = _perp_distance(p0, d, true_pos) < r_hit
            if sel.sum() < min_hits:
                continue
            t = cherenkov_hit_times(p0, d, true_pos[sel]) + noise[k, sel]
            ang_a.append(opening_angle_deg(line_fit_direction(true_pos[sel], t), d))
            ang_b.append(opening_angle_deg(line_fit_direction(nom_pos[sel], t), d))
        out[speed] = {
            "mode_a_median_deg": float(np.median(ang_a)) if ang_a else float("nan"),
            "mode_b_median_deg": float(np.median(ang_b)) if ang_b else float("nan"),
            "n_events": len(ang_a),
            "max_displacement_m": float(disp.horizontal_offsets.max()),
        }
    return out


def main() -> None:
    geo = NominalGeometry.load("arca")
    speeds = [0.0, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15]
    res = proxy_study(geo, speeds)
    print("Proxy reconstruction study (line fit, synthetic muons)")
    print(f"{'speed':>7} {'max disp':>9} {'mode a (deg)':>13} {'mode b (deg)':>13}")
    for U, r in res.items():
        print(f"{U:7.2f} {r['max_displacement_m']:9.1f} "
              f"{r['mode_a_median_deg']:13.2f} {r['mode_b_median_deg']:13.2f}")
    print("\nmode a = reco knows true displaced positions (intrinsic effect)")
    print("mode b = reco fed nominal positions (calibration-lag, the headline)")


if __name__ == "__main__":
    main()
