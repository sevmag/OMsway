# Displacement of KM3NeT optical modules under sea currents — design document

This is the authoritative description of the project: motivation, the physical
model and its derivation, parameters, the nominal geometry, the "water setting"
specification, how generated configurations plug into the NuBench / Prometheus /
GraphNeT loop, the software architecture, the validation strategy, and the phased
plan. The top-level `README.md` is a short overview; this document is the detail.

---

## 1. Motivation

A KM3NeT detection unit (DU) is not a rigid mast. It is a slender, near-
inextensible string (two 4 mm Dyneema ropes) anchored to the seabed and held
vertical only by a buoyant top float plus the buoyant glass optical modules
(DOMs). A horizontal sea current exerts hydrodynamic drag on the ropes, modules,
and buoy; with nothing rigid to resist it, the string **bends** — the anchor stays
fixed and every module above it is pushed downstream, by an amount that grows with
height and scales with the square of the current speed.

This is a real, continuous phenomenon: KM3NeT operates an acoustic positioning
system plus a compass/tiltmeter (AHRS) in every DOM and re-fits the line shape
every ~10 minutes precisely because the modules move and must be known to
< 10 cm. For **ARCA** (the long ~690 m strings) the top of a string sways by
*tens of meters* in a strong current.

Reconstruction of neutrino direction and energy relies on photon arrival *times*
at *known* sensor positions. In water light travels ~0.22 m/ns, so a 1 m position
error corresponds to a ~4–5 ns timing error — comparable to or larger than the PMT
timing resolution. Simulation/reconstruction benchmarks such as **NuBench**
(arXiv:2511.13111) treat the geometry as static and perfectly known. The bend is
therefore an unmodeled, systematic source of reconstruction error.

**Goal.** Model the bend physically, generate realistic displaced ARCA geometries
parameterized by a sea-current profile, and quantify (NuBench-style) how
reconstruction degrades with displacement.

---

## 2. Scope and decisions

| Decision | Choice |
|---|---|
| Target detector | **KM3NeT/ARCA** (= NuBench "Flower L"); large effect, large-angle regime |
| Configuration type | **Depth-resolved snapshots** (no stochastic time series) |
| Water setting | A current profile `u(z) = (speed(z), azimuth(z))`; **direction settable and depth-varying** (directional shear) |
| Purpose | **ML / simulation robustness**, in the style of NuBench |
| Study modes | **(a) intrinsic** and **(b) calibration-lag** (see §7) |

---

## 3. Physical model

### 3.1 Setup

Parameterize one DU's rope by arc length `s ∈ [0, L]`, with `s = 0` at the seabed
anchor and `s = L` at the top buoy. The shape is the curve

```
p(s) = (x(s), y(s), z(s)),    p(0) = (anchor_x, anchor_y, z_anchor),
```

with unit tangent `t(s) = dp/ds`, `|t| = 1`. The tilt (zenith) angle from vertical
is `theta(s)`, so `dz/ds = cos(theta)` and the horizontal step rate is
`sin(theta)`. The string is treated as **inextensible** (arc length, not height,
is conserved — this matters for ARCA, where the top can drop by a few meters as it
leans), and the current's vertical component is neglected for the shape.

### 3.2 Static force balance

Consider the portion of the string **above** arc length `s`. In static
equilibrium three forces act on it:

- tension from the material below, `-T(s) t(s)` (along the rope, pulling the upper
  part down toward the cut);
- net buoyancy of the upper part, `V(s) z_hat`, where
  `V(s) = sum over elements above s of W_j`  (`W_j` = buoyancy − weight, upward
  positive; DOMs and buoy positive, ropes ~slightly negative);
- total hydrodynamic drag of the upper part, the horizontal vector
  `H(s) = sum over elements above s of F_j`.

Equilibrium `-T(s) t(s) + V(s) z_hat + H(s) = 0` gives

```
T(s) t(s) = H(s) + V(s) z_hat
=>  tan(theta(s)) = |H(s)| / V(s),     T(s) = |H(s) + V(s) z_hat|
```

and the **shape ODE**

```
dp/ds = t(s) = ( sin(theta(s)) * Hhat(s),  cos(theta(s)) ),
   with  Hhat(s) = H(s)/|H(s)|  (horizontal unit vector),
         sin(theta) = |H| / sqrt(|H|^2 + V^2),
         cos(theta) = V   / sqrt(|H|^2 + V^2).
```

Integrated from the anchor upward, `p(s)` is the displaced shape; module `k` sits
at its fixed arc length `s_k` and its displaced position is `p(s_k)`. Because `H`
is a **2-D horizontal vector**, a current whose direction changes with depth makes
the string curve out of a single vertical plane — directional shear is handled with
no special case.

### 3.3 Drag

Each element contributes quadratic drag in the direction of the local current:

```
F_j = 0.5 * C_d,j * rho * A_j * |u(z_j)| * u_vec(z_j)        (vector; magnitude ~ u^2)
    = f_j * |u(z_j)| * uhat(z_j),     f_j = 0.5 * C_d,j * rho * A_j.
```

`rho ≈ 1025 kg/m^3`. DOMs and the buoy are point drags; the rope is distributed
(`f' ds` per element). `H(s)` is the cumulative vector sum of `F_j` over all
elements above `s`, evaluated at each element's **actual depth** `z_j = z(s_j)`.

### 3.4 Depth ↔ shape coupling

For a **uniform** current, `H(s)` and `V(s)` are cumulative sums of fixed per-
element constants (top-down), so the shape follows from a single bottom-up
integration — no iteration. For a **depth-varying** current, the drag at each
element depends on `z(s_j)`, which depends on the (unknown) shape. Solve by
**fixed-point iteration**: start vertical (`z(s_j) = z_anchor + s_j`), compute
`H, V` and the shape, update the element depths, and repeat. At ARCA tilts this
converges in a few iterations.

### 3.5 Small-angle reduction (validation cross-check)

If net buoyancy is dominated by the top buoy so vertical tension is nearly
constant `V(s) ≈ B`, and drag per unit length `w` is uniform, then
`tan(theta(s)) ≈ w (L−s)/B`, giving a **parabola**

```
r(s) = (w/B) (L s − s^2/2),     r(top) = w L^2 / (2 B)   ~ v^2, ~ L^2.
```

This is the ANTARES Eq. 4.5–4.10 closed form in the small-angle limit and is used
to check the numerical integrator. ARCA's large tilt (top deflection ~100 m over
~690 m, `tan(theta) ~ 0.14`) is beyond strict small-angle validity, so the full
`tan(theta)` integration above is the production model; the parabola is only a
limit check.

---

## 4. Parameters

Mechanical constants enter as `f_j` (drag) and `W_j` (net buoyancy) per element
type. KM3NeT's own fitted constants are **not published** (the Sensors 2020 paper
cites internal references), so we derive them from KM3NeT ARCA hardware and
cross-check against ANTARES's published Table 1.

### 4.1 ARCA hardware (calibrated, `km3disp/parameters.py`)

| Element | Quantity | Net buoyancy `W` | Drag inputs |
|---|---|---|---|
| DOM (17" Vitrovex sphere) | 18 per DU | **+250 N each** | d = 0.44 m, A ≈ π·0.22² ≈ 0.152 m², C_d(sphere) ≈ 0.5–1.0 |
| Strength ropes (Dyneema) | 2 × 4 mm, ~690 m | ~neutral (slightly negative) | d_eff from 2×0.004 m + 0.007 m VEOC tube, C_d(cyl) ≈ 1.0–1.2 |
| Top buoy (syntactic foam) | 1 | **+1.35 kN** | area TBD; ANTARES proxy f_buoy = 453 N·s²/m² |
| **Total net buoyancy** | per DU | **~+3.5 kN** | — |

### 4.2 ANTARES Table 1 (published proxy, arXiv:1202.3894)

Lumped constants `f_j` (with `F = f v²`) and `W_j`: storey f=383.8 N·s²/m²,
W=+265.6 N; 12 m cable f=222, W=−52.9 N; 100 m cable f=1850, W=−440 N; buoy
f=453, W≈+7 kN. These are rescaled to ARCA geometry (0.44 m DOMs, 4 mm ropes,
1.35 kN buoy) where KM3NeT values are unavailable.

Modeling choices (the two genuinely uncertain inputs): DOMs are taken as
**near-neutral** (`w_dom ≈ 0`) — the glass-sphere buoyancy roughly cancels the
module + titanium-collar weight, leaving the top buoy as the dominant restoring
tension (net `V(0) ≈ 1.0 kN`) — and the strength member is treated as a **cable
bundle** of ~0.05 m projected width/length. These plus a single calibrated
`drag_scale` define the constant set.

### 4.3 Vertical layout

18 DOMs at 36 m spacing → 612 m instrumented span; total DU height ~690 m, with
the anchor→first-DOM gap = 40 m and the top-DOM→buoy gap = 38 m (`parameters.py`,
approximate).

### 4.4 Calibration and validation (Phase 3, done)

The overall drag/buoyancy scale is fixed by the one published ARCA figure: a
700 m DU under a uniform 0.15 m/s current deflects its top buoy by ~100 m
(arXiv:2007.16090). A bisection (`calibrate_arca`) yields **`drag_scale = 0.644`**
(implied effective cable `C_d ≈ 0.77`, a benign adjustment), reproducing 100 m at
0.15 m/s by construction.

Independent validation comes from **ANTARES**, where both the constants (Table 1)
and the deflections are published. The integrator with ANTARES's own constants
gives **1.24 m at 7 cm/s** (published < 2 m) and **10.2 m at 20 cm/s** (published
~15 m; the ~30% gap reflects the published formula's storey-smearing vs the exact
point-load treatment here). The calibrated ARCA regime then falls out:

| current | top-DOM deflection | max tilt |
|---|---|---|
| 0.02 m/s (typical ARCA) | 1.8 m | 0.3° |
| 0.03 m/s (typical ARCA) | 4.0 m | 0.7° |
| 0.05 m/s | 11.0 m | 2.0° |
| 0.10 m/s | 43.8 m | 8.0° |
| 0.15 m/s | 96.7 m | 17.6° |

with displacement scaling as `v^2` (ratio 4.00 on doubling speed). Checks live in
`tests/test_parameters.py` and the report in `km3disp/validate.py`.

---

## 5. Nominal geometry (Phase 1, implemented)

Loaded from the GraphNeT/NuBench table `flower_l.parquet`:

- 115 strings × 18 DOMs = **2070 sensors**; every string vertical (constant x,y).
- Identical flat-seabed layout on all strings: deepest DOM **−3500 m**, top
  **−2888 m**, exactly **36 m** spacing.
- `sensor_id` runs bottom→top within a string; median string pitch **78.5 m**
  (Fibonacci-spiral "flower" layout), footprint radius up to **500 m**.

A DU therefore reduces to an anchor `(x, y)` footprint plus 18 module depths,
which is exactly the mechanics input. The anchor `(x, y)` equals the (constant)
string footprint; the anchor depth (seabed) sits below −3500 m by the
anchor→first-DOM segment length.

---

## 6. Water-setting specification (Phase 4)

A "water setting" is a depth-resolved current profile

```
u(z) = ( speed(z), azimuth(z) )    [horizontal only],
```

implemented as composable templates:

- **uniform** — constant speed and direction (the KM3NeT operational assumption);
- **sheared speed** — linear / two-layer `speed(z)`;
- **rotating azimuth** — `azimuth(z)` turning with depth (directional shear);
- **benthic** — a near-bottom boundary-layer modifier.

The whole 115-DU array sees the same horizontal field (the current's horizontal
decorrelation scale greatly exceeds the ~1 km array footprint); anchors are fixed
and the tops sway coherently.

---

## 7. Integration with NuBench / Prometheus / GraphNeT

NuBench already supplies the whole loop: ARCA exists as **Flower L**, Prometheus
simulates photons on it, GraphNeT (DynEdge etc.) reconstructs, and results are
scored by median opening angle, energy bias/σ, vertex resolution (m),
inelasticity, and topology AUC. NuBench assumes **static, perfectly-known**
geometry — that is the gap this project fills.

### 7.1 Two physically distinct effects

- **(a) Intrinsic.** The detector *is* displaced and reconstruction *knows* the
  true displaced positions. Tests whether the swayed layout is inherently worse
  (corridors, lever arms). Re-simulate on the displaced geometry; reco uses
  matching positions.
- **(b) Calibration-lag (headline).** The detector is displaced but reconstruction
  is fed the *nominal* positions — i.e. each pulse carries the nominal DOM
  coordinate while its photon arrival *time* reflects the true displaced geometry.
  This is the realistic failure mode (calibration lags the current) and almost
  certainly the larger effect. Mechanically clean: simulate on the displaced
  geometry, then relabel each hit's position to nominal, so GraphNeT's exact-match
  geometry lookup stays valid against the nominal table.

### 7.2 Integration contract

The generator emits two **numerically identical** artifacts (the GraphNeT
`DataRepresentation` does an exact-float-match lookup of hit xyz against the
geometry-table index, so the two sides must agree bit-for-bit):

1. **GraphNeT geometry parquet** — same schema as `flower_l.parquet`:
   `MultiIndex ['x','y','z']` plus columns `sensor_pos_x, sensor_pos_y,
   sensor_pos_z, sensor_string_id (float), sensor_id (int), t (int, 0)`.
2. **Prometheus geometry** — the displaced per-sensor position array used to drive
   the simulation, built from the same source array as artifact 1.

A thin `Detector` subclass of `FlowerL` pointing `geometry_table_path` at the
displaced parquet is the one-line reconstruction-side hook.

---

## 8. Software architecture

Package `km3disp` (pure numpy/pandas/pyarrow; no scipy — the integrators and
geometry helpers are plain numpy). Runs in the dedicated `environment.yml` conda
env (`mamba activate km3disp`).

```
km3disp/
  geometry.py    nominal ARCA -> DetectionUnit / NominalGeometry        [Phase 1, done]
  mechanics.py   arc-length 3-D force-balance integrator (one DU)        [Phase 2, done]
  parameters.py  ARCA hardware constants + calibration + ANTARES model   [Phase 3, done]
  validate.py    ANTARES cross-check, ARCA calibration + regime report   [Phase 3, done]
  currents.py    u(z) = (speed(z), azimuth(z)) profile templates         [Phase 4, done]
  displace.py    apply a current field to all 115 DUs                    [Phase 5, done]
  io.py          emit GraphNeT parquet + bit-synced Prometheus geometry  [Phase 5, done]
  study.py       displacement sweep + GraphNeT hook + proxy reco study   [Phase 6, done]
```

---

## 9. Validation strategy

1. **Small-angle limit** — numerical integrator vs the parabola `r = wL²/2B` at low
   current; must agree as `theta -> 0`.
2. **Absolute scale** — reproduce the ~100 m top deflection of a 700 m DU at
   0.15 m/s (arXiv:2007.16090).
3. **Scaling laws** — confirm `r ~ v²` and the buoyancy/length dependence.
4. **Sanity** — anchors fixed, arc length conserved, monotonic deflection with
   height, coherent downstream lean; visual "string dance" plots.

---

## 10. Phased plan and status

1. **Geometry + scaffold** — load nominal ARCA, expose detection units. **(done)**
2. **Mechanics core** — arc-length 3-D integrator for one DU; uniform current.
   **(done)** `mechanics.py`: `DUModel`/`DUConstants`/`DUShape`, pure-numpy
   quadrature with fixed-point depth coupling; validated in `tests/test_mechanics.py`
   (small-angle parabola, v² scaling, arc-length conservation, directional shear).
3. **Parameters + validation** — KM3NeT hardware constants; benchmark + small-angle
   cross-check. **(done)** `parameters.py`/`validate.py`: ANTARES reproduced from
   its own published constants (1.2 m @ 7 cm/s, 10 m @ 20 cm/s); ARCA calibrated to
   the benchmark (`drag_scale = 0.644`); realistic regime a few m at typical ARCA
   currents; checks in `tests/test_parameters.py`.
4. **Current profiles** — `u(z)` templates incl. rotating azimuth. **(done)**
   `currents.py`: `CurrentProfile` (piecewise-linear, component-interpolated) +
   `uniform`/`sheared_speed`/`rotating_azimuth`/`two_layer`/`benthic`.
5. **Array displacement + I/O** — displace 115 DUs; emit synced artifacts.
   **(done)** `displace.py` (`displace_arca` → `DisplacedArray`) and `io.py`; the
   displaced parquet matches `flower_l.parquet` schema and is bit-identical to the
   Prometheus-side file (`tests/test_displace.py`).
6. **NuBench study** — simulate → reconstruct → score vs displacement, modes (a)/(b).
   **(done)** `study.py`: the full-pipeline harness (`displacement_sweep`,
   `make_displaced_detector` GraphNeT hook, `relabel_hits_to_nominal` mode-(b)
   transform) plus a runnable numpy proxy (synthetic muons, Cherenkov hit times,
   line-fit reco). Proxy result: mode (b) median angular error rises 2.2°→4.1° as
   strings displace 0→~97 m, while mode (a) stays ~flat. The full Prometheus +
   GraphNeT/DynEdge run (heavy, not executed here) drops into the same harness.

---

## 11. References

- ANTARES Positioning System, JINST 7 (2012) T08002, **arXiv:1202.3894** — master
  line-shape model (Eqs. 4.1–4.10) and Table 1 of calibrated drag/weight constants.
- KM3NeT, *Monitoring and Reconstruction of the Shape of the Detection Units*,
  Sensors 2020, 20, 5116, **PMC7571249** — KM3NeT formalism (Eqs. 2–5), 18 DOMs,
  current speed+direction as free fit parameters.
- KM3NeT, *Deep-sea deployment by self-unrolling*, **arXiv:2007.16090** — DU
  hardware and the ~100 m @ 0.15 m/s deflection benchmark.
- KM3NeT, *DU Line Fit reconstruction*, **arXiv:2109.04914** — AHRS + acoustic
  trilateration into the line-fit model.
- **NuBench**, **arXiv:2511.13111** (JINST, DOI 10.1088/1748-0221/21/05/T05001),
  github.com/graphnet-team/NuBench — benchmark loop, geometries, metrics.
- Currents: van Haren et al. 2011 (**arXiv:1111.6482**); Tamburini/Musumeci et al.
  2017 (**PMC5362963**).
