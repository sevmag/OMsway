"""Static arc-length force-balance solver for current-driven string displacement.

A string is a rope from its seabed anchor (arc length ``s=0``) to the buoy
(``s=L``). At every cut ``s`` the part above is in static equilibrium under the
tension there, the net buoyancy ``V(s)`` of everything above (upward), and the
horizontal drag ``H(s)`` of everything above; the unit tangent is
``(H_x, H_y, V)/|.|`` and the shape is its integral along ``s``. Drag is
quadratic, ``F = f |u| u`` with ``f = 0.5 c_w rho A``. Modules (optical modules
and the buoy) are point elements; the cable is distributed. The current depends
on each element's displaced position, so the shape is found by fixed-point
iteration from the straight string.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .currents import CurrentModel
from .geometry import Geometry, String


def _cumtrapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative trapezoid of ``y`` over ``x`` with a leading zero (same length)."""
    step = np.diff(x)
    incr = 0.5 * (y[1:] + y[:-1]) * (step if y.ndim == 1 else step[:, None])
    return np.concatenate([np.zeros((1,) + y.shape[1:]), np.cumsum(incr, axis=0)])


def _above(values: np.ndarray, arc: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Sum of point ``values`` (sorted by arc length ``arc``) at or above each ``s``.

    ``side="left"`` counts an element sitting exactly on a node (the top buoy at
    ``s=L``) as above it, so the endpoint keeps a non-degenerate force balance.
    """
    rev = np.zeros((len(values) + 1,) + values.shape[1:])
    rev[:-1] = np.cumsum(values[::-1], axis=0)[::-1]
    return rev[np.searchsorted(arc, s, side="left")]


def _interp(query: np.ndarray, s: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Interpolate the ``(n, 3)`` rope shape at the arc lengths ``query``."""
    return np.column_stack([np.interp(query, s, positions[:, k]) for k in range(3)])


@dataclass
class StringShape:
    """Displaced shape of one string under a current."""

    string_id: int
    s: np.ndarray  # (n_nodes,) arc-length grid
    positions: np.ndarray  # (n_nodes, 3) rope shape, anchor to buoy
    module_positions: np.ndarray  # (n_modules, 3) displaced, in string.modules order
    converged: bool
    n_iter: int


class Solver:
    """Fixed-point arc-length solver for string displacement under a current.

    ``drag_scale`` multiplies every drag factor to absorb the joint cable-drag and
    buoyancy uncertainty (calibrated against a benchmark deflection).
    """

    def __init__(
        self,
        *,
        rho: float = 1025.0,
        drag_scale: float = 1.0,
        n_nodes: int = 2001,
        max_iter: int = 100,
        tol: float = 1e-5,
    ):
        self.rho = rho
        self.drag_scale = drag_scale
        self.n_nodes = n_nodes
        self.max_iter = max_iter
        self.tol = tol

    def solve(
        self,
        geometry: Geometry,
        current: CurrentModel,
        *,
        time: float = 0.0,
        apply: bool = True,
    ) -> list[StringShape]:
        """Displace every string; each samples the current at its own position.

        With ``apply`` (the default) the displaced positions are written back onto
        the modules, so the geometry then reports the displaced state
        (``positions``/``displacements``/``to_prometheus_geo``). Arc lengths are
        read from the geometry's nominal baseline, so re-solving stays correct.
        """
        shapes: list[StringShape] = []
        nominal = geometry.unperturbed_positions
        i = 0
        for string in geometry:
            n = string.n_modules
            shape = self.solve_string(string, current, nominal=nominal[i : i + n], time=time)
            i += n
            if apply:
                for m, p in zip(string.modules, shape.module_positions):
                    m.position = p
            shapes.append(shape)
        return shapes

    def solve_string(
        self,
        string: String,
        current: CurrentModel,
        *,
        nominal: np.ndarray | None = None,
        time: float = 0.0,
    ) -> StringShape:
        anchor = string.anchor
        # Arc length and rope length come from the nominal (undisplaced) layout, so
        # a solve stays valid even after displaced positions are written back.
        ref = string.positions() if nominal is None else nominal
        arc = np.linalg.norm(ref - anchor, axis=1)
        length = float(arc.max())
        s = np.linspace(0.0, length, self.n_nodes)

        order = np.argsort(arc)
        arc = arc[order]
        f_pt = self.drag_scale * np.array([m.drag_factor(self.rho) for m in string.modules])[order]
        w_pt = np.array([m.buoyancy for m in string.modules])[order]

        f_rope = self.drag_scale * string.cable.drag_factor(self.rho)
        w_rope = string.cable.buoyancy_per_length

        # Net upward force above each node is current-independent.
        vertical = _above(w_pt, arc, s) + w_rope * (length - s)

        positions = anchor + np.column_stack([np.zeros_like(s), np.zeros_like(s), s])
        converged, n_iter = False, 0
        for n_iter in range(1, self.max_iter + 1):
            u = current(positions, time)[:, :2]
            drag_lin = (f_rope * np.linalg.norm(u, axis=1))[:, None] * u
            cum = _cumtrapz(drag_lin, s)
            horizontal = cum[-1] - cum  # rope drag above each node

            elem = _interp(arc, s, positions)
            u_elem = current(elem, time)[:, :2]
            drag_pt = (f_pt * np.linalg.norm(u_elem, axis=1))[:, None] * u_elem
            horizontal = horizontal + _above(drag_pt, arc, s)

            force = np.column_stack([horizontal, vertical])
            tangent = force / np.maximum(np.linalg.norm(force, axis=1), 1e-12)[:, None]
            new = anchor + _cumtrapz(tangent, s)
            converged = bool(np.abs(new - positions).max() < self.tol)
            positions = new
            if converged:
                break

        module_positions = np.empty((len(string.modules), 3))
        module_positions[order] = _interp(arc, s, positions)
        return StringShape(
            string_id=string.string_id,
            s=s,
            positions=positions,
            module_positions=module_positions,
            converged=converged,
            n_iter=n_iter,
        )
