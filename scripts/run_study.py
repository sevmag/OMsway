"""Run the Phase 6 displacement-reconstruction proxy study and save its outputs.

Reproducible entry point: ``python scripts/run_study.py``. Writes the swept
results to ``data/study/proxy_results.csv`` and the figure to
``docs/figures/phase6_degradation.png``. The numbers come from
``km3disp.study.proxy_study`` (synthetic muons, Cherenkov hit times, line-fit
reconstruction); see docs/DESIGN.md section 7.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Put the project root on the path so the script runs without an installed
# package (matches how the tests resolve km3disp via conftest.py).
sys.path.insert(0, str(ROOT))

import matplotlib
import pandas as pd

matplotlib.use("Agg")  # headless: render straight to file
import matplotlib.pyplot as plt

from km3disp.geometry import NominalGeometry
from km3disp.study import proxy_study
SPEEDS = [0.0, 0.02, 0.03, 0.05, 0.07, 0.10, 0.13, 0.15]
N_EVENTS = 800
SEED = 7


def main() -> None:
    geo = NominalGeometry.load("arca")
    res = proxy_study(geo, SPEEDS, n_events=N_EVENTS, seed=SEED)

    df = pd.DataFrame(
        [
            {
                "speed_m_s": U,
                "max_displacement_m": res[U]["max_displacement_m"],
                "mode_a_intrinsic_deg": res[U]["mode_a_median_deg"],
                "mode_b_calibration_lag_deg": res[U]["mode_b_median_deg"],
                "n_events": res[U]["n_events"],
            }
            for U in SPEEDS
        ]
    )
    csv_path = ROOT / "data" / "study" / "proxy_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(df.to_string(index=False))
    print(f"\nwrote {csv_path}")

    _plot(df, ROOT / "docs" / "figures" / "phase6_degradation.png")


def _plot(df: pd.DataFrame, out: Path) -> None:
    disp = df["max_displacement_m"].to_numpy()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(disp, df["mode_a_intrinsic_deg"], "o-",
            label="mode (a) intrinsic — reco knows true positions")
    ax.plot(disp, df["mode_b_calibration_lag_deg"], "s-", color="C3",
            label="mode (b) calibration-lag — reco fed nominal")
    ax.axvspan(0, 4.0, color="green", alpha=0.12, label="typical ARCA (<=3 cm/s)")
    for _, row in df.iterrows():
        if row["speed_m_s"] in (0.05, 0.10, 0.15):
            ax.annotate(f"{row['speed_m_s']:.2f} m/s",
                        (float(row["max_displacement_m"]),
                         float(row["mode_b_calibration_lag_deg"])),
                        textcoords="offset points", xytext=(4, 5), fontsize=8)
    ax.set_xlabel("max string displacement [m]")
    ax.set_ylabel("median reconstruction error [deg]")
    ax.set_title("ARCA reconstruction degradation vs displacement\n"
                 "(proxy: line-fit on synthetic muons)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
