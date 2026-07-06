# art

Decorative renders — **not scientific examples.** The animations here use
stylized, exaggerated, non-physical currents chosen to look good, and are not
validated against anything. For the actual physics and validation, see
[`../scripts/`](../scripts).

- `sway_animation.py` — generates `sway.gif`: ARCA's strings swaying under an
  invented traveling-wave current. Needs `plotly` + `kaleido` (kaleido drives a
  headless Chromium, so the 60-frame render wants a few GB of memory). Run:
  `python art/sway_animation.py [out.gif] [arca.geo]`.
- `sway.gif` — the rendered animation (used in the top-level README).
