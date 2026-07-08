"""Hierarchical KM3NeT/IceCube detector geometry.

A :class:`Geometry` is a collection of :class:`String`s, each carrying an ordered
list of :class:`Module`s (a :class:`SphericalOM` or any other device). A module
holds its world ``position``; the string's seabed ``anchor`` is the fixed
reference, and a module's string-frame offset is ``position - anchor``. A module
exposes the drag coefficient ``c_w``, frontal ``area``, and net ``buoyancy`` the
solver needs as abstract properties, so each device is free to store or compute
them; :class:`SphericalOM` derives ``area`` from a radius. Each string also holds
a :class:`Cable` giving the distributed rope material's drag and buoyancy per unit
length.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

import numpy as np

# Column schema of the GraphNeT/NuBench per-sensor geometry tables.
XYZ = ["sensor_pos_x", "sensor_pos_y", "sensor_pos_z"]
STRING_COL = "sensor_string_id"
SENSOR_COL = "sensor_id"

# Known geometry tables (vertical-string detectors share this loader).
GEOMETRY_TABLE_DIR = Path(
    "/n/home12/smagel/data/graphnet/data/geometry_tables"
)
NAMED_GEOMETRIES = {
    "arca": GEOMETRY_TABLE_DIR / "nubench" / "flower_l.parquet",
    "orca": GEOMETRY_TABLE_DIR / "nubench" / "flower_s.parquet",
}


class Module(ABC):
    """A single device on a string: an optical module, a buoy, or anything else.

    A module is a point element -- an identifier plus its world ``position``
    ``(x, y, z)``. Its offset within the string frame is ``position - anchor``,
    derived by the string where needed.

    The mechanical constants the solver needs are abstract properties, leaving
    each concrete device free to store or compute them (e.g. from a radius): net
    ``buoyancy`` (up positive), drag coefficient ``c_w``, and frontal ``area``,
    which enter the drag as ``F = 0.5 c_w rho A |u| u``.

    A module also carries its orientation as two independent degrees of freedom:
    ``axis`` is the unit vector its symmetry axis points along (upright
    ``(0, 0, 1)`` by default), and ``torsion`` is the roll about that axis
    (radians). The sway solver sets ``axis`` from the local rope tangent (the
    tilt); ``torsion`` (the DOM heading/yaw) is supplied separately, since a
    current exerts no torque about the axis of a symmetric module. ``tilt`` and
    ``tilt_azimuth`` expose ``axis`` as angles.
    """

    def __init__(
        self,
        module_id: int,
        position: np.ndarray,
        *,
        axis: np.ndarray | None = None,
        torsion: float = 0.0,
    ):
        self.module_id = module_id
        self.position = np.asarray(position, float)
        self.axis = np.array([0.0, 0.0, 1.0]) if axis is None else self._unit(axis)
        self.torsion = float(torsion)

    @staticmethod
    def _unit(v: np.ndarray) -> np.ndarray:
        """Return ``v`` normalised to a unit vector; error on the zero vector."""
        v = np.asarray(v, float)
        norm = np.linalg.norm(v)
        if norm == 0.0:
            raise ValueError("axis must be a non-zero vector")
        return v / norm

    @staticmethod
    def axis_from_angles(tilt: float, azimuth: float) -> np.ndarray:
        """Unit axis for a polar ``tilt`` from vertical leaning toward ``azimuth`` (radians)."""
        s = np.sin(tilt)
        return np.array([s * np.cos(azimuth), s * np.sin(azimuth), np.cos(tilt)])

    @property
    def tilt(self) -> float:
        """Polar tilt of ``axis`` away from vertical (+z), radians."""
        return float(np.arccos(np.clip(self.axis[2], -1.0, 1.0)))

    @property
    def tilt_azimuth(self) -> float:
        """Azimuth of the lean (from +x toward +y), radians; ``0`` when upright."""
        return float(np.arctan2(self.axis[1], self.axis[0]))

    def set_orientation(
        self,
        *,
        tilt: float | None = None,
        tilt_azimuth: float | None = None,
        torsion: float | None = None,
    ) -> None:
        """Set orientation from angles (radians).

        ``tilt`` and ``tilt_azimuth`` set ``axis`` and must be given together;
        ``torsion`` sets the roll. Any argument left ``None`` is unchanged.
        """
        if (tilt is None) != (tilt_azimuth is None):
            raise ValueError("tilt and tilt_azimuth must be set together")
        if tilt is not None and tilt_azimuth is not None:
            self.axis = self.axis_from_angles(tilt, tilt_azimuth)
        if torsion is not None:
            self.torsion = float(torsion)

    @property
    @abstractmethod
    def kind(self) -> str:
        """Short label of the device type, e.g. ``'OM'``, ``'buoy'``."""

    @property
    @abstractmethod
    def buoyancy(self) -> float:
        """Net buoyancy (buoyancy minus weight, up positive), newtons."""

    @property
    @abstractmethod
    def c_w(self) -> float:
        """Drag coefficient (dimensionless)."""

    @property
    @abstractmethod
    def area(self) -> float:
        """Frontal area presented to the flow, m^2."""

    def drag_factor(self, rho: float) -> float:
        """``f = 0.5 c_w rho A`` (N s^2 / m^2), so ``|F| = f |u|^2``."""
        return 0.5 * self.c_w * rho * self.area

    def __repr__(self) -> str:
        return f"{type(self).__name__}(id={self.module_id!r}, position={self.position.tolist()})"


class SphericalOM(Module):
    """A KM3NeT optical module modelled as a sphere.

    The frontal ``area`` follows from the glass sphere's ``radius`` and ``c_w`` is
    the smooth-sphere drag coefficient; the net ``buoyancy`` is supplied per
    module.
    """

    _SPHERE_CD = 0.47  # smooth-sphere drag coefficient, subcritical Reynolds

    def __init__(
        self,
        module_id: int,
        position: np.ndarray,
        radius: float,
        buoyancy: float,
        *,
        axis: np.ndarray | None = None,
        torsion: float = 0.0,
    ):
        super().__init__(module_id, position, axis=axis, torsion=torsion)
        self.radius = float(radius)
        self._buoyancy = float(buoyancy)

    @property
    def kind(self) -> str:
        return "OM"

    @property
    def buoyancy(self) -> float:
        return self._buoyancy

    @property
    def c_w(self) -> float:
        return self._SPHERE_CD

    @property
    def area(self) -> float:
        return np.pi * self.radius**2


class Buoy(Module):
    """The buoyancy element at the top of a string.

    A buoy is not modelled as a sphere, so its net ``buoyancy``, drag coefficient
    ``c_w``, and frontal ``area`` are supplied directly. Placement is optional, so
    a bare ``Buoy`` acts as a template a loader clones and repositions onto each
    string; an unplaced buoy has ``module_id`` ``-1``.
    """

    def __init__(
        self,
        buoyancy: float,
        c_w: float,
        area: float,
        *,
        module_id: int = -1,
        position: np.ndarray | None = None,
    ):
        super().__init__(module_id, np.zeros(3) if position is None else position)
        self._buoyancy = float(buoyancy)
        self._c_w = float(c_w)
        self._area = float(area)

    @property
    def kind(self) -> str:
        return "buoy"

    @property
    def buoyancy(self) -> float:
        return self._buoyancy

    @property
    def c_w(self) -> float:
        return self._c_w

    @property
    def area(self) -> float:
        return self._area


class Cable(ABC):
    """The distributed string material (ropes, backbone tube, ...) of a string.

    Unlike the point modules, a cable contributes to the string balance
    continuously along its length. Its drag per unit length enters as
    ``0.5 c_w rho w |u| u`` (``w`` the projected width), and
    ``buoyancy_per_length`` is the net upward force per metre. Concrete cables
    provide these constants however they like (e.g. from a diameter).
    """

    @property
    @abstractmethod
    def c_w(self) -> float:
        """Drag coefficient (dimensionless)."""

    @property
    @abstractmethod
    def width(self) -> float:
        """Projected width presented to the flow, per unit length (m)."""

    @property
    @abstractmethod
    def buoyancy_per_length(self) -> float:
        """Net buoyancy per unit length (up positive), N/m."""

    def drag_factor(self, rho: float) -> float:
        """Per-unit-length drag factor ``0.5 c_w rho w`` (N s^2 / m^3)."""
        return 0.5 * self.c_w * rho * self.width

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(c_w={self.c_w:.3g}, width={self.width:.3g}, "
            f"buoyancy_per_length={self.buoyancy_per_length:.3g})"
        )


class CylindricalCable(Cable):
    """A cable of circular cross-section: the drag width is its ``diameter``.

    ``c_w`` defaults to the crossflow drag coefficient of a long cylinder; the net
    ``buoyancy_per_length`` is supplied directly.
    """

    _CYLINDER_CD = 1.2  # crossflow drag coefficient of a long cylinder

    def __init__(
        self, diameter: float, buoyancy_per_length: float, *, c_w: float | None = None
    ):
        self.diameter = float(diameter)
        self._buoyancy_per_length = float(buoyancy_per_length)
        self._c_w = self._CYLINDER_CD if c_w is None else float(c_w)

    @property
    def c_w(self) -> float:
        return self._c_w

    @property
    def width(self) -> float:
        return self.diameter

    @property
    def buoyancy_per_length(self) -> float:
        return self._buoyancy_per_length


class String:
    """One string: an anchor footprint carrying an ordered list of modules.

    ``anchor`` is the string's fixed seabed reference in the geometry frame; each
    module holds its own world ``position``, and :meth:`offsets` gives them
    relative to the anchor.
    """

    def __init__(
        self,
        string_id: int,
        anchor: np.ndarray,
        modules: Sequence[Module],
        cable: Cable,
    ):
        self.string_id = string_id
        self.anchor = np.asarray(anchor, float)
        self.modules = list(modules)
        self.cable = cable

    def __len__(self) -> int:
        return len(self.modules)

    def __iter__(self):
        return iter(self.modules)

    def __getitem__(self, module_id: int) -> Module:
        m = next((m for m in self.modules if m.module_id == module_id), None)
        if m is None:
            raise KeyError(module_id)
        return m

    @property
    def n_modules(self) -> int:
        return len(self.modules)

    @property
    def module_ids(self) -> np.ndarray:
        return np.array([m.module_id for m in self.modules])

    def positions(self) -> np.ndarray:
        """``(n, 3)`` world positions of the modules."""
        if not self.modules:
            return np.empty((0, 3))
        return np.array([m.position for m in self.modules])

    def offsets(self) -> np.ndarray:
        """``(n, 3)`` module positions relative to the anchor (string frame)."""
        return self.positions() - self.anchor

    @property
    def rope_length(self) -> float:
        """Arc length from the anchor to the top of the string (its highest module).

        In the nominal straight configuration this is the anchor-to-buoy distance.
        """
        if not self.modules:
            return 0.0
        top = max(self.modules, key=lambda m: m.position[2])
        return float(np.linalg.norm(top.position - self.anchor))

    def __repr__(self) -> str:
        return f"String(id={self.string_id!r}, {self.n_modules} modules)"


class Geometry:
    """A full detector geometry: a collection of strings.

    At construction the module positions are captured as the unperturbed baseline
    (:attr:`unperturbed_positions`), so a later displaced state can be measured
    against it with :meth:`displacements`.
    """

    def __init__(self, strings: list[String]):
        self.strings = list(strings)
        self.unperturbed_positions = self.positions()  # nominal baseline snapshot

    @classmethod
    def from_prometheus_geo(
        cls,
        path: str | Path,
        *,
        buoy: Module,
        buoy_gap: float,
        cable: Cable,
        z_floor: float | None = None,
        buoyancy: float = 0.0,
        radius: float | None = None,
    ) -> "Geometry":
        """Build a geometry from a Prometheus ``.geo`` file.

        The file carries a metadata block (medium, DOM radius) then a
        ``### Modules ###`` marker followed by tab-separated
        ``x y z string_id module_number`` rows. Optical modules load as
        :class:`SphericalOM` with the file's DOM radius (``radius`` overrides it,
        in metres) and net ``buoyancy`` (near-neutral by default, since the file
        carries no buoyancy).

        A geo file has no buoy, so ``buoy`` (any initialised :class:`Module`) is
        cloned onto each string ``buoy_gap`` metres above its top module. Optical
        modules keep file-order ids matching ``sensor_id``; the per-string buoys
        take ids after them. A geo file also has no rope material, so ``cable`` (a
        :class:`Cable`) is shared by every string.

        A string's anchor is its footprint ``(mean x, mean y)`` at the seabed
        ``z_floor``. The geo file carries no floor, so ``z_floor`` is supplied here
        (absolute depth, m); when omitted, each string anchors at its own deepest
        module, i.e. a zero seabed gap.
        """
        path = Path(path)
        lines = path.read_text().splitlines()
        try:
            start = next(
                i for i, ln in enumerate(lines) if ln.strip() == "### Modules ###"
            )
        except StopIteration:
            raise ValueError(f"{path}: no '### Modules ###' marker found")

        if radius is None:
            for ln in lines[:start]:
                if "radius" in ln.lower():
                    radius = float(ln.split()[-1]) / 100.0  # metadata gives cm
                    break
            if radius is None:
                raise ValueError(f"{path}: DOM radius absent from metadata; pass radius=")

        by_string: dict[int, list[tuple[int, np.ndarray]]] = {}
        module_id = 0
        for ln in lines[start + 1 :]:
            if not ln.strip():
                continue
            c = ln.split("\t")
            pos = np.array([float(c[0]), float(c[1]), float(c[2])])
            by_string.setdefault(int(c[3]), []).append((module_id, pos))
            module_id += 1

        strings = []
        buoy_id = module_id  # buoys follow the optical-module ids
        for string_id, entries in by_string.items():
            xyz = np.array([pos for _, pos in entries])
            floor = xyz[:, 2].min() if z_floor is None else z_floor
            anchor = np.array([xyz[:, 0].mean(), xyz[:, 1].mean(), floor])
            oms = [
                SphericalOM(mid, pos, radius=radius, buoyancy=buoyancy)
                for mid, pos in entries
            ]
            top = max(oms, key=lambda m: m.position[2])
            b = copy.copy(buoy)
            b.module_id = buoy_id
            b.position = top.position + np.array([0.0, 0.0, buoy_gap])
            buoy_id += 1
            strings.append(String(string_id, anchor, [*oms, b], cable))
        return cls(strings)

    def to_prometheus_geo(self, path: str | Path, *, medium: str = "water") -> Path:
        """Write the current module positions as a Prometheus ``.geo`` file.

        The inverse of :meth:`from_prometheus_geo`: a ``### Metadata ###`` block
        (medium, DOM radius from the modules) then tab-separated
        ``x y z string_id module_number`` rows. Only sensors are written -- buoys
        are excluded, as they are not in a geo file -- and current positions are
        used, so a displaced geometry writes its displaced ``.geo``.
        """
        lines = ["### Metadata ###", f"Medium:\t{medium}"]
        radius = next(
            (m.radius for s in self.strings for m in s if isinstance(m, SphericalOM)),
            None,
        )
        if radius is not None:
            lines.append(f"DOM Radius [cm]:\t{radius * 100:g}")
        lines.append("### Modules ###")
        for s in self.strings:
            n = 0
            for m in s:
                if m.kind == "buoy":
                    continue
                x, y, z = m.position
                lines.append(f"{x}\t{y}\t{z}\t{s.string_id}\t{n}")
                n += 1
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n")
        return path

    def __len__(self) -> int:
        return len(self.strings)

    def __iter__(self):
        return iter(self.strings)

    def __getitem__(self, string_id: int) -> String:
        s = next((s for s in self.strings if s.string_id == string_id), None)
        if s is None:
            raise KeyError(string_id)
        return s

    @property
    def n_strings(self) -> int:
        return len(self.strings)

    @property
    def n_modules(self) -> int:
        return sum(s.n_modules for s in self.strings)

    def positions(self) -> np.ndarray:
        """``(N, 3)`` world positions of every module in the detector."""
        if not self.strings:
            return np.empty((0, 3))
        return np.vstack([s.positions() for s in self.strings])

    def module_ids(self) -> np.ndarray:
        """Module ids aligned row-for-row with :meth:`positions`."""
        if not self.strings:
            return np.array([], dtype=int)
        return np.concatenate([s.module_ids for s in self.strings])

    def string_ids(self) -> np.ndarray:
        """String id per module, aligned row-for-row with :meth:`positions`."""
        if not self.strings:
            return np.array([], dtype=int)
        return np.concatenate(
            [np.full(s.n_modules, s.string_id) for s in self.strings]
        )

    def displacements(self) -> np.ndarray:
        """Current module positions minus the unperturbed baseline (``(N, 3)``)."""
        return self.positions() - self.unperturbed_positions

    def __repr__(self) -> str:
        return f"Geometry({self.n_strings} strings, {self.n_modules} modules)"
