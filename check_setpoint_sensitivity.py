"""
Sensitivity check: does the forecaster actually respond to the setpoint knob?

This is the gate for M5. The optimizer works by asking the forecaster a
counterfactual -- "if I hold the CRAC setpoint at X for the next H minutes,
where do the racks end up and what does cooling cost?" -- and then picking the
best answer. If the model's predictions barely move when X moves, there is
nothing to optimize and the optimizer would just return noise.

So: take ONE episode of real plant history, then re-predict it under a sweep of
hypothetical setpoints via build_features(setpoint_override=...). Everything
except the setpoint columns stays fixed, so any change in the prediction is
attributable to the knob.

Note on what stays fixed: the anchor features (current temps, lags, load) come
from the episode as actually run. That is deliberate and matches how the
optimizer will be used -- it observes the plant's real current state, then asks
what happens if it changes the setpoint FROM HERE.

Run:
    python3 check_setpoint_sensitivity.py

Prints a sweep table + a verdict, and writes setpoint_sensitivity.png.
"""

from __future__ import annotations
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DEFAULT
from dataset import run_episode, build_features, HORIZONS_MIN
from forecaster import TwinForecaster

SWEEP = np.arange(16.0, 24.5, 1.0)     # candidate setpoints to test, °C
EPISODE_SEED = 999                     # unseen by training (which used seed=0)
# Minimum credible response. Physics: cooling gain is 8 kW/K, so 1 K of setpoint
# is ~8 kW of cooling heat withheld. Over an 8 K sweep the hottest rack should
# move by well over a degree. Anything flatter means the knob is being ignored.
MIN_SPAN_C = 0.5


def main() -> int:
    try:
        fc = TwinForecaster.load("forecaster.joblib")
    except FileNotFoundError:
        print("No forecaster.joblib found. Run train_forecaster.py first.")
        return 1

    print("Generating one held-out episode (seed=%d)..." % EPISODE_SEED)
    rng = np.random.default_rng(EPISODE_SEED)
    series = run_episode(DEFAULT, rng)

    # Trim the tail the same way make_supervised does, so no row references a
    # future that ran off the end of the episode.
    valid = len(series["t_rack_max"]) - max(HORIZONS_MIN) * 6
    it_kw = series["it_power_kw"][:valid]

    # Sanity: the override must not change the feature schema, or the model's
    # column indices would silently point at the wrong things.
    _, names_ref = build_features(series, setpoint_override=SWEEP[0])
    if names_ref != fc.feature_names:
        print("FEATURE SCHEMA MISMATCH -- forecaster.joblib is stale.")
        print("  model expects:", fc.feature_names)
        print("  dataset gives:", names_ref)
        print("Retrain with train_forecaster.py.")
        return 1

    results = {h: {"rack": [], "cool": [], "pue": []} for h in HORIZONS_MIN}

    for sp in SWEEP:
        X, _ = build_features(series, setpoint_override=float(sp))
        X = X[:valid]
        for h in HORIZONS_MIN:
            rack = fc.predict(X, "t_rack_max", h)
            cool = fc.predict(X, "cooling_elec_kw", h)
            # PUE = (IT + cooling elec) / IT, using the episode's real IT load.
            pue = (it_kw + cool) / it_kw
            results[h]["rack"].append(float(np.mean(rack)))
            results[h]["cool"].append(float(np.mean(cool)))
            results[h]["pue"].append(float(np.mean(pue)))

    _print_tables(results)
    verdict = _print_verdict(results)
    _plot(results)
    print("\nSaved plot -> setpoint_sensitivity.png")
    return 0 if verdict else 2


def _print_tables(results: dict) -> None:
    limit = DEFAULT.rack.limit_temp_c
    for h in HORIZONS_MIN:
        r = results[h]
        print(f"\n{h}-minute horizon   (mean over {len(SWEEP)} setpoints, "
              f"rack limit {limit:.0f} °C)")
        print(f"  {'setpoint':>9} | {'pred rack':>10} | {'margin':>8} | "
              f"{'pred cool':>10} | {'pred PUE':>9}")
        print("  " + "-" * 60)
        for i, sp in enumerate(SWEEP):
            rack = r["rack"][i]
            flag = "  <-- OVER LIMIT" if rack >= limit else ""
            print(f"  {sp:>7.1f}°C | {rack:>8.2f}°C | {limit - rack:>+7.2f} | "
                  f"{r['cool'][i]:>8.2f}kW | {r['pue'][i]:>9.4f}{flag}")


def _print_verdict(results: dict) -> bool:
    print("\n" + "=" * 68)
    print("VERDICT")
    print("=" * 68)

    ok = True
    for h in HORIZONS_MIN:
        rack = np.array(results[h]["rack"])
        cool = np.array(results[h]["cool"])
        span = rack.max() - rack.min()
        # Slope in °C of rack temp per °C of setpoint, via least squares.
        slope = np.polyfit(SWEEP, rack, 1)[0]
        # Monotone rising? Warmer supply air should mean warmer racks.
        mono = bool(np.all(np.diff(rack) > -1e-9))
        # Cooling should fall as the setpoint rises.
        cool_falls = bool(np.all(np.diff(cool) < 1e-9))

        print(f"\n  {h}-minute horizon")
        print(f"    rack temp span over sweep : {span:.3f} °C "
              f"({'ok' if span >= MIN_SPAN_C else 'TOO FLAT'})")
        print(f"    slope d(rack)/d(setpoint) : {slope:+.3f} °C/°C")
        print(f"    rack rises with setpoint  : {'yes' if mono else 'NO'}")
        print(f"    cooling falls w/ setpoint : {'yes' if cool_falls else 'NO'}")

        if span < MIN_SPAN_C or not mono or not cool_falls:
            ok = False

    print()
    if ok:
        print("  PASS -- the model responds to the knob in the right direction.")
        print("  There is a real tradeoff for the optimizer to exploit.")
    else:
        print("  FAIL -- the model is not usable for optimization yet.")
        print("  Likely causes: setpoint columns missing from the feature set,")
        print("  forecaster.joblib trained before they were added, or the")
        print("  training setpoints never varied.")
    return ok


def _plot(results: dict) -> None:
    limit = DEFAULT.rack.limit_temp_c
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Setpoint sensitivity — does the forecaster respond to the knob?",
                 fontsize=14, fontweight="bold")

    ax = axes[0]
    for h in HORIZONS_MIN:
        ax.plot(SWEEP, results[h]["rack"], "o-", label=f"{h} min")
    ax.axhline(limit, ls="--", color="k", lw=1, label="rack limit")
    ax.set_title("Predicted hottest rack")
    ax.set_xlabel("CRAC setpoint (°C)"); ax.set_ylabel("°C")
    ax.legend(fontsize=8)

    ax = axes[1]
    for h in HORIZONS_MIN:
        ax.plot(SWEEP, results[h]["cool"], "o-", label=f"{h} min")
    ax.set_title("Predicted cooling electricity")
    ax.set_xlabel("CRAC setpoint (°C)"); ax.set_ylabel("kW")
    ax.legend(fontsize=8)

    ax = axes[2]
    for h in HORIZONS_MIN:
        ax.plot(SWEEP, results[h]["pue"], "o-", label=f"{h} min")
    ax.set_title("Implied PUE")
    ax.set_xlabel("CRAC setpoint (°C)"); ax.set_ylabel("PUE")
    ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig("setpoint_sensitivity.png", dpi=115)


if __name__ == "__main__":
    sys.exit(main())
