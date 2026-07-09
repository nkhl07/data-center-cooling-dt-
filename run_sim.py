"""
Milestone 1 runner: drive the RC plant, save a CSV of sensor readings,
and produce a validation plot.

Run:  python run_sim.py
Outputs:  sim_output.csv, validation_plot.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DEFAULT
from thermal_model import DataCenterThermalModel, make_load_profile


def run(cfg=DEFAULT):
    model = DataCenterThermalModel(cfg)
    load = make_load_profile(cfg)
    n_steps = load.shape[0]
    dt = cfg.sim.dt_s

    rows = []
    for k in range(n_steps):
        obs = model.step(load[k])
        rows.append({
            "t_min": k * dt / 60.0,
            "it_power_kw": obs["it_power_kw"],
            "T_air": obs["T_air"],
            "T_rack_max": obs["T_rack_max"],
            "q_cooling_kw": obs["q_cooling_kw"],
            "cooling_elec_kw": obs["cooling_elec_kw"],
            "total_facility_kw": obs["total_facility_kw"],
            "pue": obs["pue"],
        })
    return pd.DataFrame(rows)


def validate(df, cfg=DEFAULT):
    """Sanity checks that the physics behaves correctly."""
    checks = []

    # 1) Rising IT load should raise air temperature (positive correlation).
    corr = np.corrcoef(df["it_power_kw"], df["T_air"])[0, 1]
    checks.append(("load-temp correlation > 0.5", corr > 0.5, f"corr={corr:.2f}"))

    # 2) Cooling electricity should rise when air temp rises.
    corr_c = np.corrcoef(df["T_air"], df["cooling_elec_kw"])[0, 1]
    checks.append(("warmer air -> more cooling", corr_c > 0.5, f"corr={corr_c:.2f}"))

    # 3) PUE must always be >= 1.0 (cooling only adds overhead).
    pue_ok = df["pue"].min() >= 1.0
    checks.append(("PUE >= 1.0 always", bool(pue_ok), f"min={df['pue'].min():.3f}"))

    # 4) System should stay bounded (no runaway) within a reasonable band.
    bounded = df["T_air"].max() < 60 and df["T_rack_max"].max() < 70
    checks.append(("temps stay bounded", bool(bounded),
                   f"Tair_max={df['T_air'].max():.1f}, Track_max={df['T_rack_max'].max():.1f}"))

    print("\nValidation checks")
    print("-" * 50)
    all_ok = True
    for name, ok, detail in checks:
        flag = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"[{flag}] {name:32s} {detail}")
    print("-" * 50)
    print("PUE range: {:.3f} - {:.3f}  (mean {:.3f})".format(
        df["pue"].min(), df["pue"].max(), df["pue"].mean()))
    print("Rack thermal limit: {:.1f} C | peak rack temp reached: {:.1f} C".format(
        cfg.rack.limit_temp_c, df["T_rack_max"].max()))
    print("Overall:", "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return all_ok


def plot(df, path="validation_plot.png", cfg=DEFAULT):
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(df["t_min"], df["it_power_kw"], color="tab:blue")
    axes[0].set_ylabel("IT load (kW)")
    axes[0].set_title("Data Center RC Thermal Model - Milestone 1 validation")

    axes[1].plot(df["t_min"], df["T_air"], label="Room air", color="tab:orange")
    axes[1].plot(df["t_min"], df["T_rack_max"], label="Hottest rack", color="tab:red")
    axes[1].axhline(cfg.rack.limit_temp_c, ls="--", color="k", lw=1, label="Rack limit")
    axes[1].set_ylabel("Temp (C)")
    axes[1].legend(loc="upper left")

    ax2 = axes[2]
    ax2.plot(df["t_min"], df["cooling_elec_kw"], color="tab:green", label="Cooling elec (kW)")
    ax2.set_ylabel("Cooling elec (kW)")
    ax2.set_xlabel("Time (min)")
    ax3 = ax2.twinx()
    ax3.plot(df["t_min"], df["pue"], color="tab:purple", label="PUE")
    ax3.set_ylabel("PUE")
    lines = ax2.get_lines() + ax3.get_lines()
    ax2.legend(lines, [l.get_label() for l in lines], loc="upper left")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"\nSaved plot -> {path}")


if __name__ == "__main__":
    df = run()
    df.to_csv("sim_output.csv", index=False)
    print(f"Wrote sim_output.csv ({len(df)} rows)")
    validate(df)
    plot(df)
