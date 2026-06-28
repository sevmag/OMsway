"""Arc-length 3-D force-balance model for a single detection unit.

The string is parameterized by arc length ``s`` from the seabed anchor (s=0) to the
top buoy (s=L). At each ``s`` the portion of the string above is in static
equilibrium under the tension at the cut, the net buoyancy ``V(s)`` of everything
above (upward), and the horizontal drag ``H(s)`` of everything above. The unit
tangent is therefore ``t = (H_x, H_y, V) / |(H_x, H_y, V)|`` and the shape is the
integral of the tangent along ``s``. See ``docs/DESIGN.md`` section 3.

Drag is quadratic in the local current, ``F = f * |u| * u`` with
``f = 0.5 * C_d * rho * A``; net buoyancy ``W`` is positive upward. DOMs and the
buoy are point elements; the rope is distributed. When the current varies with
depth, the drag at each element depends on that element's (unknown) depth, so the
shape is found by fixed-point iteration starting from the vertical string.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Union

import numpy as np

# A current field: maps an array of depths z (m) to horizontal velocity vectors
# (m/s), shape (len(z), 2) = (u_x, u_y).
CurrentField = Callable[[np.ndarray], np.ndarray]

# A rope property (drag or net-buoyancy per unit length) is either uniform
# (scalar) or a function of arc length s, so ANTARES-style distinct bottom
# cables and tapered profiles are expressible.
RopeParam = Union[float, Callable[[np.ndarray], np.ndarray]]


def _resolve_rope(val: RopeParam, s: np.ndarray) -> np.ndarray:
    """Evaluate a rope property to a per-node array over the arc-length grid."""
    raw = val(s) if callable(val) else val
    return np.broadcast_to(np.asarray(raw, float), s.shape).astype(float)


def _cumtrapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative trapezoid of ``y`` over ``x`` with a leading zero (same length)."""
    dx = np.diff(x)
    avg = 0.5 * (y[1:] + y[:-1])
    incr = avg * dx if y.ndim == 1 else avg * dx[:, None]
    cum = np.cumsum(incr, axis=0)
    zero = np.zeros((1,) + y.shape[1:])
    return np.concatenate([zero, cum], axis=0)


def _rev_cumtrapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Reverse cumulative trapezoid: ``integral_{s}^{x_max} y`` at each node."""
    cum = _cumtrapz(y, x)
    return cum[-1] - cum


@dataclass
class DUConstants:
    """Per-element drag (``f = 0.5 C_d rho A``) and net buoyancy (``W``, up>0).

    Rope terms are per unit length; DOM and buoy terms are per element. Drag enters
    as ``F = f |u| u`` so that ``|F| = f |u|^2``.
    """

    f_dom: float  # N s^2 / m^2, per DOM
    w_dom: float  # N, per DOM (buoyancy - weight)
    f_rope: RopeParam  # N s^2 / m^3, per unit rope length (scalar or f(s))
    w_rope: RopeParam  # N / m, per unit rope length (scalar or f(s))
    f_buoy: float  # N s^2 / m^2, top buoy
    w_buoy: float  # N, top buoy


@dataclass
class DUShape:
    """Result of solving one detection unit's shape under a current field."""

    s: np.ndarray  # (n_nodes,) arc-length grid
    positions: np.ndarray  # (n_nodes, 3) shape along the rope
    dom_positions: np.ndarray  # (n_dom, 3) displaced module positions
    tilt: np.ndarray  # (n_nodes,) zenith angle from vertical, radians
    tension: np.ndarray  # (n_nodes,) rope tension, N
    anchor: np.ndarray  # (3,)
    n_iter: int
    converged: bool

    @property
    def top_position(self) -> np.ndarray:
        return self.positions[-1]

    @property
    def buoy_deflection(self) -> float:
        """Horizontal offset of the top buoy from the anchor."""
        return float(np.hypot(*(self.positions[-1, :2] - self.anchor[:2])))

    @property
    def dom_horizontal_offsets(self) -> np.ndarray:
        """Per-DOM horizontal offset from the anchor footprint."""
        return np.hypot(*(self.dom_positions[:, :2] - self.anchor[:2]).T)

    @property
    def top_dom_deflection(self) -> float:
        return float(self.dom_horizontal_offsets[-1])

    @property
    def max_tilt_deg(self) -> float:
        return float(np.degrees(self.tilt.max()))


@dataclass
class DUModel:
    """A detection unit: anchor, module arc-lengths, rope length, and constants."""

    anchor: np.ndarray  # (3,) seabed anchor (x, y, z)
    dom_arclengths: np.ndarray  # (n_dom,) arc length of each DOM above the anchor
    rope_length: float  # L, anchor to top buoy
    constants: DUConstants

    @classmethod
    def from_unit(
        cls,
        unit,
        *,
        seabed_gap: float,
        buoy_gap: float,
        constants: DUConstants,
    ) -> "DUModel":
        """Build from a nominal ``DetectionUnit`` plus the two end-segment lengths.

        ``seabed_gap`` is the anchor->first-DOM distance, ``buoy_gap`` the
        top-DOM->buoy distance. In the nominal vertical state arc length equals
        height above the anchor, so the module arc-lengths follow directly.
        """
        anchor_z = unit.z_bottom - seabed_gap
        dom_s = unit.dom_z - anchor_z
        anchor = np.array([unit.anchor_xy[0], unit.anchor_xy[1], anchor_z])
        return cls(
            anchor=anchor,
            dom_arclengths=dom_s,
            rope_length=float(dom_s[-1] + buoy_gap),
            constants=constants,
        )

    def solve(
        self,
        current: CurrentField,
        *,
        n_nodes: int = 4001,
        max_iter: int = 100,
        tol: float = 1e-5,
    ) -> DUShape:
        c = self.constants
        s = np.linspace(0.0, self.rope_length, n_nodes)
        ax, ay, az = self.anchor
        dom_s = np.asarray(self.dom_arclengths, float)
        n_dom = len(dom_s)

        # Net buoyancy above each node from point elements is analytic: the count
        # of DOMs whose arc-length exceeds the node, plus the buoy. The rope term
        # is the net buoyancy of the rope above the node; both are current-
        # independent, so V is fixed across the depth-coupling iteration.
        n_above = n_dom - np.searchsorted(dom_s, s, side="right")
        V_pts = n_above * c.w_dom + c.w_buoy
        f_rope = _resolve_rope(c.f_rope, s)
        V_rope = _rev_cumtrapz(_resolve_rope(c.w_rope, s), s)
        V = V_pts + V_rope

        z = az + s  # vertical-string initial guess
        positions = np.column_stack([np.full(n_nodes, ax), np.full(n_nodes, ay), z])
        H = np.zeros((n_nodes, 2))
        converged = False
        n_iter = 0
        for n_iter in range(1, max_iter + 1):
            u = np.atleast_2d(current(z))
            drag_lin = (f_rope * np.linalg.norm(u, axis=1))[:, None] * u
            H_rope = _rev_cumtrapz(drag_lin, s)

            z_dom = np.interp(dom_s, s, z)
            u_dom = np.atleast_2d(current(z_dom))
            drag_dom = c.f_dom * np.linalg.norm(u_dom, axis=1)[:, None] * u_dom
            rev = np.zeros((n_dom + 1, 2))
            if n_dom:
                rev[:-1] = np.cumsum(drag_dom[::-1], axis=0)[::-1]
            idx = np.searchsorted(dom_s, s, side="right")

            z_buoy = float(np.interp(self.rope_length, s, z))
            u_buoy = np.atleast_2d(current(np.array([z_buoy])))[0]
            drag_buoy = c.f_buoy * float(np.linalg.norm(u_buoy)) * u_buoy

            H = H_rope + rev[idx] + drag_buoy

            force = np.column_stack([H[:, 0], H[:, 1], V])
            T = np.maximum(np.linalg.norm(force, axis=1), 1e-12)
            tangent = force / T[:, None]

            new = np.empty_like(positions)
            new[:, 0] = ax + _cumtrapz(tangent[:, 0], s)
            new[:, 1] = ay + _cumtrapz(tangent[:, 1], s)
            new[:, 2] = az + _cumtrapz(tangent[:, 2], s)
            dz = float(np.max(np.abs(new[:, 2] - z)))
            positions = new
            z = positions[:, 2]
            if dz < tol:
                converged = True
                break

        dom_positions = np.column_stack(
            [np.interp(dom_s, s, positions[:, k]) for k in range(3)]
        )
        tilt = np.arctan2(np.linalg.norm(H, axis=1), V)
        tension = np.linalg.norm(np.column_stack([H[:, 0], H[:, 1], V]), axis=1)
        return DUShape(
            s=s,
            positions=positions,
            dom_positions=dom_positions,
            tilt=tilt,
            tension=tension,
            anchor=self.anchor.copy(),
            n_iter=n_iter,
            converged=converged,
        )


def uniform_current(speed: float, azimuth_deg: float = 0.0) -> CurrentField:
    """A depth-independent horizontal current of given speed (m/s) and azimuth."""
    a = np.radians(azimuth_deg)
    v = speed * np.array([np.cos(a), np.sin(a)])
    return lambda z: np.tile(v, (np.size(z), 1))
