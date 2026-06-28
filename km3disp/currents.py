"""Depth-resolved current profiles (the "water settings").

A current field maps depth ``z`` (m, negative downward) to a horizontal velocity
vector ``(u_x, u_y)`` in m/s, shape ``(len(z), 2)`` -- the ``CurrentField``
signature the mechanics solver consumes. Profiles are specified by speed and
azimuth as functions of depth; both may vary with depth, so directional shear (a
current that rotates with depth) is expressible. A profile stores velocity
COMPONENTS at depth nodes and interpolates those, which avoids azimuth
wrap-around and keeps the field smooth; outside the node range the edge values
are held, since the abyssal current does not grow beyond the instrumented span.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mechanics import CurrentField, uniform_current

__all__ = [
    "CurrentProfile",
    "uniform",
    "sheared_speed",
    "rotating_azimuth",
    "two_layer",
    "benthic",
]


@dataclass
class CurrentProfile:
    """Piecewise-linear depth profile of the horizontal current.

    ``z_nodes`` ascending (deepest first); ``vx``/``vy`` the velocity components
    at those depths. Components, not (speed, azimuth), are interpolated.
    """

    z_nodes: np.ndarray
    vx: np.ndarray
    vy: np.ndarray

    def __call__(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, float)
        ux = np.interp(z, self.z_nodes, self.vx)
        uy = np.interp(z, self.z_nodes, self.vy)
        return np.column_stack([ux, uy])

    @classmethod
    def from_speed_azimuth(cls, z_nodes, speed, azimuth_deg) -> "CurrentProfile":
        z = np.asarray(z_nodes, float)
        order = np.argsort(z)
        z = z[order]
        spd = np.broadcast_to(np.asarray(speed, float), z.shape)[order]
        az = np.radians(np.broadcast_to(np.asarray(azimuth_deg, float), z.shape))[order]
        return cls(z, spd * np.cos(az), spd * np.sin(az))

    @property
    def speed(self) -> np.ndarray:
        return np.hypot(self.vx, self.vy)

    @property
    def azimuth_deg(self) -> np.ndarray:
        return np.degrees(np.arctan2(self.vy, self.vx))


def uniform(speed: float, azimuth_deg: float = 0.0) -> CurrentField:
    """Depth-independent current (the KM3NeT operational assumption)."""
    return uniform_current(speed, azimuth_deg)


def sheared_speed(
    z_bottom: float,
    z_top: float,
    speed_bottom: float,
    speed_top: float,
    azimuth_deg: float = 0.0,
) -> CurrentProfile:
    """Speed varying linearly with depth at fixed direction."""
    return CurrentProfile.from_speed_azimuth(
        [z_bottom, z_top], [speed_bottom, speed_top], azimuth_deg
    )


def rotating_azimuth(
    z_bottom: float,
    z_top: float,
    speed: float,
    azimuth_bottom_deg: float,
    azimuth_top_deg: float,
    n_nodes: int = 16,
) -> CurrentProfile:
    """Direction rotating with depth (directional shear) at (optionally) fixed speed.

    Sampled at ``n_nodes`` depths so the component interpolation follows the
    rotation smoothly rather than chording across a large angle.
    """
    z = np.linspace(z_bottom, z_top, n_nodes)
    frac = (z - z_bottom) / (z_top - z_bottom)
    az = azimuth_bottom_deg + frac * (azimuth_top_deg - azimuth_bottom_deg)
    spd = np.broadcast_to(np.asarray(speed, float), z.shape)
    return CurrentProfile.from_speed_azimuth(z, spd, az)


def two_layer(
    z_interface: float,
    lower_speed: float,
    lower_azimuth_deg: float,
    upper_speed: float,
    upper_azimuth_deg: float,
    *,
    blend: float = 5.0,
) -> CurrentProfile:
    """Two depth layers with a thin linear blend across the interface."""
    z = [z_interface - blend, z_interface + blend]
    spd = [lower_speed, upper_speed]
    az = [lower_azimuth_deg, upper_azimuth_deg]
    return CurrentProfile.from_speed_azimuth(z, spd, az)


def benthic(
    base: CurrentField,
    seabed_z: float,
    layer_thickness: float,
    reduction: float = 0.5,
) -> CurrentField:
    """Wrap a current field with a near-bottom frictional speed reduction.

    Within ``layer_thickness`` above ``seabed_z`` the speed is scaled from
    ``reduction`` (at the bed) up to 1 (at the top of the layer).
    """

    def field(z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, float)
        u = np.atleast_2d(base(z))
        height = np.clip((z - seabed_z) / layer_thickness, 0.0, 1.0)
        scale = reduction + (1.0 - reduction) * height
        return u * scale[:, None]

    return field
