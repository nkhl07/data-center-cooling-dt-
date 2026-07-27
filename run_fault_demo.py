"""
Fault-detection demo (Milestone 4).

Run:
    python3 run_fault_demo.py

Steps:
  1. Load the trained forecaster (train_forecaster.py must have run first).
  2. Calibrate the detector on a NORMAL run (learn how big residuals usually are).
  3. Run a FAULTED episode with the same workload, injecting a CRAC capacity
     loss partway through.
  4. Detect the fault from prediction residuals; report detection latency.
  5. Plot temperature + predictions and the anomaly signal.

Produces fault_detection.png.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DEFAULT
from thermal_model import make_load_profile
from faults import Fault, simulate
from forecaster import TwinForecaster
from anomaly import compute_residuals, ResidualDetector, detection_latency_min


def load_forecaster():
    try:
        return TwinForecaster.load("forecaster.joblib")
    except FileNotFoundError:
        print("forecaster.joblib not found -- training it now...")
        from dataset import make_supervised, train_test_split_by_episode
        data = make_supervised(n_episodes=12, seed=0)
        tr, _ = train_test_split_by_episode(data)
        fc = TwinForecaster().fit(data["X"], data["y"], data["feature_names"], tr)
        fc.save("forecaster.joblib")
        return fc


def main():
    cfg = DEFAULT
    fc = load_forecaster()
    load = make_load_profile(cfg, seed=42)         # same workload for both runs

    # --- calibrate on a normal (fault-free) run ---
    normal_series, _ = simulate(cfg, load=load, fault=Fault(kind="none"))
    normal_resid, _ = compute_residuals(normal_series, fc, "t_rack_max", horizon_min=5)
    detector = ResidualDetector(k=5.0, debounce=12).fit(normal_resid)

    # --- faulted run: cooler loses 45% capacity from t=180 to 260 min ---
    fault = Fault(kind="crac_degradation", start_min=180, end_min=260, magnitude=0.45)
    series, active = simulate(cfg, load=load, fault=fault)
    resid, aligned = compute_residuals(series, fc, "t_rack_max", horizon_min=5)
    flags, z = detector.flags(resid)

    lat = detection_latency_min(active, flags, cfg.sim.dt_s)
    n = len(series["t_rack_max"])
    t = np.arange(n) * cfg.sim.dt_s / 60.0

    # false positives before the fault window
    pre = t < fault.start_min
    fp = int(flags[pre].sum())

    print("\nFault-detection demo")
    print("-" * 46)
    print(f"Injected: CRAC -{fault.magnitude*100:.0f}% capacity, "
          f"{fault.start_min:.0f}-{fault.end_min:.0f} min")
    print(f"Detection latency: {lat:.1f} min" if lat is not None else "NOT DETECTED")
    print(f"False alarms before fault: {fp} steps")
    print(f"Peak |z| during fault: {np.nanmax(np.abs(z[active])):.1f}")

    _plot(t, series, aligned, z, flags, active, detector, fault, lat)
    print("Saved plot -> fault_detection.png")


def _plot(t, series, aligned, z, flags, active, det, fault, lat):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Milestone 4 — fault injection & anomaly detection",
                 fontsize=15, fontweight="bold")

    fs, fe = fault.start_min, fault.end_min
    for ax in (ax1, ax2):
        ax.axvspan(fs, fe, color="orange", alpha=0.15, label="fault active")

    ax1.plot(t, series["t_rack_max"], color="tab:red", label="actual rack temp")
    ax1.plot(t, aligned, color="tab:blue", lw=1, alpha=0.8,
             label="forecast (5 min, normal model)")
    ax1.axhline(DEFAULT.rack.limit_temp_c, ls="--", color="k", lw=1, label="rack limit")
    fired = flags & ~np.isnan(series["t_rack_max"])
    ax1.scatter(t[fired], series["t_rack_max"][fired], s=10, color="black",
                zorder=5, label="ALARM")
    ax1.set_ylabel("°C"); ax1.legend(fontsize=8, loc="upper left")
    ax1.set_title("Reality diverges from the normal-model forecast when the cooler fails")

    ax2.plot(t, z, color="tab:purple", lw=1, label="residual z-score")
    ax2.axhline(det.k, ls="--", color="tab:red", lw=1, label=f"threshold (±{det.k:g})")
    ax2.axhline(-det.k, ls="--", color="tab:red", lw=1)
    ax2.set_ylabel("z-score"); ax2.set_xlabel("time (min)")
    ax2.legend(fontsize=8, loc="upper left")
    ttl = f"Anomaly signal — detected {lat:.1f} min after onset" if lat is not None \
        else "Anomaly signal"
    ax2.set_title(ttl)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("fault_detection.png", dpi=115)


if __name__ == "__main__":
    main()
