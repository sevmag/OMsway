"""Current field models (the "water settings").

A :class:`CurrentModel` maps position ``(x, y, z)`` and time to a water velocity
vector ``(u_x, u_y, u_z)`` in m/s. Concrete models range from a steady uniform
horizontal flow to a depth-resolved profile; each is a special case that ignores
some of its arguments. Depth profiles interpolate velocity COMPONENTS (not speed
and azimuth), which avoids azimuth wrap-around and keeps the field smooth; the
edge values are held outside the instrumented depth span.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

__all__ = [
    "CurrentModel",
    "UniformCurrent",
    "DepthProfileCurrent",
]


class CurrentModel(ABC):
    """A current field over space and time.

    Calling the model is the one primitive: given positions and a time it
    returns the water velocity vector at each. Speed and heading derive from it.
    The depth-only, steady, purely horizontal currents used elsewhere in this
    module are all special cases -- models that ignore some of their arguments.
    """

    @abstractmethod
    def __call__(self, position: np.ndarray, time: float = 0.0) -> np.ndarray:
        """Velocity vectors (m/s) at ``position`` and ``time``.

        ``position`` is ``(N, 3)`` Cartesian ``(x, y, z)`` in metres (``z``
        negative downward); a single ``(3,)`` point is also accepted. ``time``
        is in hours (scalar, or broadcastable to ``N``); steady models ignore
        it. Returns ``(N, 3)`` components ``(u_x, u_y, u_z)``.
        """

    def speed(self, position: np.ndarray, time: float = 0.0) -> np.ndarray:
        """Current speed (m/s): the magnitude of the velocity vector."""
        return np.linalg.norm(self(position, time), axis=-1)

    def azimuth_deg(self, position: np.ndarray, time: float = 0.0) -> np.ndarray:
        """Heading of the horizontal current, degrees CCW from +x."""
        u = self(position, time)
        return np.degrees(np.arctan2(u[..., 1], u[..., 0]))

    def _repr_params(self) -> dict[str, object]:
        """Constructor parameters to render in ``repr``; subclasses override."""
        return {}

    def __repr__(self) -> str:
        parts = (
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v!r}"
            for k, v in self._repr_params().items()
        )
        return f"{type(self).__name__}({', '.join(parts)})"


class UniformCurrent(CurrentModel):
    """A steady, purely horizontal current: one speed and heading everywhere.

    The KM3NeT operational assumption -- no vertical component and no variation
    with position or time. Speed and azimuth are resolved to a fixed velocity
    vector at construction; the inherited :meth:`speed`/:meth:`azimuth_deg`
    recover those constants.
    """

    def __init__(self, speed: float, azimuth_deg: float = 0.0):
        a = np.radians(azimuth_deg)
        self._velocity = np.array([speed * np.cos(a), speed * np.sin(a), 0.0])

    def __call__(self, position: np.ndarray, time: float = 0.0) -> np.ndarray:
        n = np.atleast_2d(np.asarray(position, float)).shape[0]
        return np.tile(self._velocity, (n, 1))

    def _repr_params(self) -> dict[str, object]:
        vx, vy, _ = self._velocity
        return {
            "speed": float(np.hypot(vx, vy)),
            "azimuth_deg": float(np.degrees(np.arctan2(vy, vx))),
        }


class DepthProfileCurrent(CurrentModel):
    """Steady current whose velocity varies with depth, uniform horizontally.

    Velocity COMPONENTS are interpolated linearly against depth ``z`` (not speed
    and azimuth, which avoids azimuth wrap-around and keeps the field smooth).
    Outside the node range the edge values are held, since the abyssal current
    is not extrapolated beyond the instrumented span.
    """

    def __init__(self, z_nodes, vx, vy, vz=0.0):
        z = np.asarray(z_nodes, float)
        order = np.argsort(z)
        self._z = z[order]
        self._v = np.column_stack(
            [np.broadcast_to(np.asarray(c, float), z.shape) for c in (vx, vy, vz)]
        )[order]

    def __call__(self, position: np.ndarray, time: float = 0.0) -> np.ndarray:
        z = np.atleast_2d(np.asarray(position, float))[:, 2]
        return np.column_stack([np.interp(z, self._z, self._v[:, k]) for k in range(3)])

    @classmethod
    def from_speed_azimuth(cls, z_nodes, speed, azimuth_deg) -> "DepthProfileCurrent":
        z = np.asarray(z_nodes, float)
        spd = np.broadcast_to(np.asarray(speed, float), z.shape)
        az = np.radians(np.broadcast_to(np.asarray(azimuth_deg, float), z.shape))
        return cls(z, spd * np.cos(az), spd * np.sin(az))

    def _repr_params(self) -> dict[str, object]:
        return {
            "n_nodes": int(self._z.size),
            "z_min": float(self._z[0]),
            "z_max": float(self._z[-1]),
        }
