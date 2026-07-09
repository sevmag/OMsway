"""Torsion (roll about the DOM axis) models.

The mooring solve fixes each optical module's position and axis (its tilt) but
says nothing about its roll about that axis: a current exerts no torque on a
symmetric module. That roll -- the DOM yaw/heading -- is a separate degree of
freedom, set at deployment and measured in situ by the module compass, not
predicted by the shape solve. A :class:`TorsionModel` assigns it as a per-string
constant (deployment heading) plus an optional yaw model plus per-module random
scatter; a concrete model overrides whichever of those it represents.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .currents import CurrentModel
from .geometry import Geometry, Module, String


class TorsionModel(ABC):
    """Assign each optical module a torsion (roll about its axis, radians).

    :meth:`apply` writes ``module.torsion`` for every optical module as
    ``per_string_constant + yaw + scatter``. Subclasses override whichever of
    those three contributions they model; the rest stay zero. Non-OM devices
    (buoys) are left untouched.
    """

    def apply(self, geometry: Geometry, current: CurrentModel) -> None:
        """Compute and store the torsion of every optical module in ``geometry``.

        Pass a solved geometry, so module positions and axes reflect the
        displaced state that a yaw model may depend on.
        """
        for string in geometry:
            constant = self.per_string_constant(string, current)
            for module in string:
                if module.kind != "OM":
                    continue
                module.torsion = (
                    constant + self.yaw(module, string, current) + self.scatter(module, string)
                )

    def per_string_constant(self, string: String, current: CurrentModel) -> float:
        """Deployment heading shared by every module on ``string`` (radians)."""
        return 0.0

    def yaw(self, module: Module, string: String, current: CurrentModel) -> float:
        """Per-module yaw-model contribution (radians)."""
        return 0.0

    @abstractmethod
    def scatter(self, module: Module, string: String) -> float:
        """Per-module random scatter about the axis (radians)."""


class RandomScatter(TorsionModel):
    """Independent zero-mean Gaussian roll per module.

    Models the residual per-module heading spread a compass sees around a
    detection unit's deployment orientation (KM3NeT reports a few degrees). The
    per-string constant and yaw contributions stay zero, so this is scatter only.
    ``sigma`` is the standard deviation in radians; ``seed`` fixes the draw.
    """

    def __init__(self, sigma: float, *, seed: int = 0):
        self.sigma = float(sigma)
        self._rng = np.random.default_rng(seed)

    def scatter(self, module: Module, string: String) -> float:
        return float(self._rng.normal(0.0, self.sigma))
