# km3net_position_change

Modeling the displacement of KM3NeT optical modules under sea currents, and
generating realistic displaced detector configurations for reconstruction
studies.

> **Full design document: [`docs/DESIGN.md`](docs/DESIGN.md)** — physics
> derivation, parameters, geometry, current-profile spec, NuBench integration,
> validation, and the phased plan in detail. This README is the short overview.

## Idea

Each KM3NeT detection unit (DU) is a slender, near-inextensible string anchored
to the seabed and held vertical by a top buoy. A horizontal sea current drags
the string sideways; because nothing rigid resists it, the string bends and the
optical modules (DOMs) are pushed downstream, by an amount that grows with
height and scales with the square of the current speed. Real DUs do this
continuously (KM3NeT tracks it live with acoustics + compasses), but
simulation/reconstruction frameworks such as NuBench treat the geometry as
static and perfectly known. This project models the bend physically and emits
displaced ARCA geometries so the reconstruction impact can be studied.

Target: **KM3NeT/ARCA** (the long ~690 m strings, where the effect reaches tens
of meters). Configurations are **depth-resolved snapshots**: the "water setting"
is a current profile `u(z) = (speed(z), azimuth(z))` whose speed and direction
may both vary with depth.

## Physical model

Static large-angle mooring-line force balance, integrated along the rope arc
length `s`:

```
H(s) = sum over elements above s of  f_j * |u(z)| * u_vec(z)     (drag, vector)
V(s) = sum over elements above s of  W_j                          (net buoyancy)
tan(theta(s)) = |H(s)| / V(s)
dp/ds = (sin(theta) * Hhat(s), cos(theta))
```

with `f_j = 0.5 * C_d * rho * A_j`. A depth-varying current direction makes `H` a
2-D vector, so the string curves in 3-D (directional shear). Current depth is
coupled to the (unknown) shape and solved by fixed-point iteration. Reference:
ANTARES line-shape model (arXiv:1202.3894), reused by KM3NeT (Sensors 2020,
PMC7571249).

## Plan / status

1. **Geometry + scaffold** — load nominal ARCA, expose detection units. *(done)*
2. **Mechanics core** — arc-length 3-D force-balance integrator for one DU. *(done)*
3. **Parameters + validation** — KM3NeT hardware constants; reproduce ~100 m top
   deflection at 0.15 m/s; ANTARES cross-check. *(done)*
4. **Current profiles** — `u(z)` profile spec + templates (uniform, sheared speed,
   rotating azimuth, benthic). *(done)*
5. **Array displacement + I/O** — displace all 115 DUs; emit GraphNeT geometry
   parquet + bit-synced Prometheus geometry. *(done)*
6. **NuBench-style study** — simulate -> reconstruct -> score vs displacement
   magnitude, modes (a) intrinsic and (b) calibration-lag. *(done — harness +
   runnable line-fit proxy; full Prometheus/DynEdge run is the remaining heavy
   step)*

## Environment

A dedicated conda env is defined in `environment.yml` (conda-forge, Python 3.12;
numpy / pandas / pyarrow / matplotlib / pytest plus an editable install of this
package):

```
mamba env create -f environment.yml
mamba activate km3disp
```

Then, from the project root:

```
python -m pytest                 # tests
python -m km3disp.validate       # Phase-3 validation report
python scripts/run_study.py      # Phase-6 study -> data/study/proxy_results.csv + figure
```

The package itself is pure numpy/pandas/pyarrow (matplotlib only for figures) and
deliberately uses no scipy, so the integrators and geometry helpers are plain
numpy. The heavyweight Phase-6 pipeline (Prometheus + GraphNeT/DynEdge) runs in
the separate graphnet env, not here.
