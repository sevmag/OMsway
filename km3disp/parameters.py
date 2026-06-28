"""Mechanical parameters for ARCA, plus an ANTARES model for validation.

ARCA constants are derived from KM3NeT hardware (DOM 0.44 m sphere, two 4 mm
Dyneema ropes + a backbone tube as a cable bundle, 1.35 kN top buoy) using
``f = 0.5 C_d rho A``. KM3NeT's own fitted constants are not published, and two
inputs are genuinely uncertain — the in-water net buoyancy budget and the
effective drag of the full cable bundle — so a single overall ``drag_scale`` is
calibrated to the one published ARCA figure (~100 m top-buoy deflection of a
~700 m DU at 0.15 m/s, arXiv:2007.16090). The mechanics itself is validated
independently against ANTARES, whose drag/buoyancy constants AND resulting
deflections are both published (arXiv:1202.3894, Table 1).
"""

from __future__ import annotations

import numpy as np

from .mechanics import DUConstants, DUModel, uniform_current

RHO = 1025.0  # kg/m^3, seawater density

# --- ARCA hardware (arXiv:2007.16090) ---
DOM_DIAMETER = 0.44  # m, 17-inch Vitrovex sphere
DOM_AREA = np.pi * (DOM_DIAMETER / 2) ** 2  # m^2, frontal area
BUOY_BUOYANCY = 1350.0  # N, syntactic-foam top buoy
CABLE_BUNDLE_WIDTH = 0.05  # m, projected width/length of 2x4 mm ropes + tube + collars

# Vertical layout: 18 DOMs at 36 m spacing (612 m span); ~690 m total DU height,
# the ~78 m balance split between the anchor->first-DOM and top-DOM->buoy spans.
ARCA_SEABED_GAP = 40.0  # m, anchor to first (deepest) DOM
ARCA_BUOY_GAP = 38.0  # m, top DOM to buoy
ARCA_BENCHMARK_SPEED = 0.15  # m/s
ARCA_BENCHMARK_DEFLECTION = 100.0  # m, top-buoy deflection of a ~700 m DU


def arca_constants(
    *,
    drag_scale: float = 1.0,
    c_d_dom: float = 0.5,
    c_d_cable: float = 1.2,
    c_d_buoy: float = 0.8,
    buoy_area: float = 0.25,
    w_dom: float = 0.0,
    w_rope: float = -0.5,
    w_buoy: float = BUOY_BUOYANCY,
) -> DUConstants:
    """Hardware-derived ARCA constants; ``drag_scale`` is the calibration knob.

    DOMs are taken as near-neutral (``w_dom~0``): the glass-sphere buoyancy
    roughly cancels the module + titanium-collar weight, leaving the top buoy as
    the dominant restoring tension. ``drag_scale`` absorbs the joint uncertainty
    in cable-bundle drag and net buoyancy and is fixed by ``calibrate_arca``.
    """
    return DUConstants(
        f_dom=drag_scale * 0.5 * c_d_dom * RHO * DOM_AREA,
        w_dom=w_dom,
        f_rope=drag_scale * 0.5 * c_d_cable * RHO * CABLE_BUNDLE_WIDTH,
        w_rope=w_rope,
        f_buoy=drag_scale * 0.5 * c_d_buoy * RHO * buoy_area,
        w_buoy=w_buoy,
    )


def arca_du_model(unit, *, constants: DUConstants | None = None, **constant_kwargs) -> DUModel:
    """Build an ARCA ``DUModel`` for a nominal ``DetectionUnit``."""
    if constants is None:
        constants = arca_constants(**constant_kwargs)
    return DUModel.from_unit(
        unit, seabed_gap=ARCA_SEABED_GAP, buoy_gap=ARCA_BUOY_GAP, constants=constants
    )


def benchmark_du_model(constants: DUConstants, *, height: float = 700.0) -> DUModel:
    """A standalone ~700 m, 18-DOM ARCA-like DU for the deflection benchmark."""
    n_dom = 18
    spacing = 36.0
    dom_s = ARCA_SEABED_GAP + spacing * np.arange(n_dom)
    return DUModel(
        anchor=np.array([0.0, 0.0, -3500.0]),
        dom_arclengths=dom_s,
        rope_length=height,
        constants=constants,
    )


def calibrate_arca(
    *,
    target_m: float = ARCA_BENCHMARK_DEFLECTION,
    speed: float = ARCA_BENCHMARK_SPEED,
    height: float = 700.0,
    tol: float = 1e-3,
    max_iter: int = 80,
    **constant_kwargs,
) -> float:
    """Find the ``drag_scale`` reproducing the benchmark top-buoy deflection.

    Deflection increases monotonically with drag, so a bisection on a wide bracket
    converges. Returns the calibrated ``drag_scale``.
    """

    def deflection(scale: float) -> float:
        consts = arca_constants(drag_scale=scale, **constant_kwargs)
        model = benchmark_du_model(consts, height=height)
        return model.solve(uniform_current(speed)).buoy_deflection - target_m

    lo, hi = 1e-3, 100.0
    f_lo = deflection(lo)
    for _ in range(max_iter):
        mid = np.sqrt(lo * hi)  # geometric bisection over the wide bracket
        f_mid = deflection(mid)
        if abs(f_mid) < tol * target_m:
            return mid
        if (f_mid > 0) == (f_lo > 0):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return np.sqrt(lo * hi)


# --- ANTARES validation model (arXiv:1202.3894, Table 1) ---
# Both the constants below and the deflections they should produce (top-storey
# < 2 m at 7 cm/s, ~15 m at 20 cm/s) are published, making this an independent
# check of the integrator. The line has a draggy 100 m bottom cable distinct from
# the inter-storey cabling, exercised via the per-arc-length rope profile.
def antares_model() -> DUModel:
    n_storey = 25
    bottom_cable = 100.0  # m, BSS -> first storey
    storey_spacing = 14.5  # m
    dom_s = bottom_cable + storey_spacing * np.arange(n_storey)
    rope_length = float(dom_s[-1] + 2.0)

    f_storey, w_storey = 383.8, 265.6
    f_buoy, w_buoy = 453.0, 7000.0
    # Table 1 lumped cable totals -> per-length over their spans.
    inter_span = dom_s[-1] - bottom_cable
    f_inter, w_inter = 222.0 / inter_span, -52.9 / inter_span
    f_bottom, w_bottom = 1850.0 / bottom_cable, -440.0 / bottom_cable

    def f_rope(s):
        return np.where(s < bottom_cable, f_bottom, f_inter)

    def w_rope(s):
        return np.where(s < bottom_cable, w_bottom, w_inter)

    constants = DUConstants(
        f_dom=f_storey, w_dom=w_storey, f_rope=f_rope, w_rope=w_rope,
        f_buoy=f_buoy, w_buoy=w_buoy,
    )
    return DUModel(
        anchor=np.array([0.0, 0.0, -2475.0]),
        dom_arclengths=dom_s,
        rope_length=rope_length,
        constants=constants,
    )
