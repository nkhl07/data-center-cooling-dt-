"""
Train and evaluate the twin forecaster (Milestone 3).

Run:
    python3 train_forecaster.py

Produces:
    forecaster.joblib          the trained models
    forecast_eval.png          predicted-vs-actual + error-by-horizon plots
and prints an accuracy table comparing the model to a persistence baseline.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import make_supervised, train_test_split_by_episode, HORIZONS_MIN
from forecaster import TwinForecaster, print_report


def main():
    print("Generating training data (varied plant episodes)...")
    data = make_supervised(n_episodes=12, seed=0)
    train_mask, test_mask = train_test_split_by_episode(data, test_frac=0.25)
    print(f"  {data['X'].shape[0]} samples, {data['X'].shape[1]} features")
    print(f"  train={train_mask.sum()}  test={test_mask.sum()}")

    print("Training one linear (Ridge) model per (target, horizon)...")
    fc = TwinForecaster().fit(
        data["X"], data["y"], data["feature_names"], train_mask
    )

    report = fc.evaluate(data["X"], data["y"], test_mask)
    print_report(report)

    fc.save("forecaster.joblib")
    print("\nSaved model -> forecaster.joblib")

    _plot(fc, data, test_mask, report)
    print("Saved plot -> forecast_eval.png")


def _plot(fc, data, test_mask, report):
    Xte = data["X"][test_mask]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Twin Forecaster — evaluation on held-out episodes",
                 fontsize=15, fontweight="bold")

    # (0,0) predicted vs actual rack temp, 30 min horizon, a slice
    h = 30
    truth = data["y"]["t_rack_max"][h][test_mask]
    pred = fc.predict(Xte, "t_rack_max", h)
    sl = slice(0, 600)
    ax = axes[0, 0]
    ax.plot(truth[sl], label="actual", color="tab:red")
    ax.plot(pred[sl], label="predicted", color="tab:blue", alpha=0.8)
    ax.set_title(f"Hottest rack temp — {h} min ahead")
    ax.set_xlabel("test timestep"); ax.set_ylabel("°C"); ax.legend(fontsize=8)

    # (0,1) scatter predicted vs actual (perfect = diagonal)
    ax = axes[0, 1]
    ax.scatter(truth, pred, s=4, alpha=0.3, color="tab:blue")
    lo, hi = truth.min(), truth.max()
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_title(f"Predicted vs actual (rack temp, {h} min)")
    ax.set_xlabel("actual °C"); ax.set_ylabel("predicted °C")

    # (1,0) error grows with horizon — rack temp
    ax = axes[1, 0]
    hs = HORIZONS_MIN
    model_mae = [report["t_rack_max"][hh]["mae"] for hh in hs]
    base_mae = [report["t_rack_max"][hh]["baseline_mae"] for hh in hs]
    ax.plot(hs, model_mae, "o-", label="ML model", color="tab:blue")
    ax.plot(hs, base_mae, "o--", label="persistence baseline", color="tab:gray")
    ax.set_title("Rack-temp error vs horizon")
    ax.set_xlabel("horizon (min)"); ax.set_ylabel("MAE (°C)"); ax.legend(fontsize=8)

    # (1,1) cooling energy predicted vs actual, 15 min
    h2 = 15
    truth2 = data["y"]["cooling_elec_kw"][h2][test_mask]
    pred2 = fc.predict(Xte, "cooling_elec_kw", h2)
    ax = axes[1, 1]
    ax.plot(truth2[sl], label="actual", color="tab:green")
    ax.plot(pred2[sl], label="predicted", color="tab:blue", alpha=0.8)
    ax.set_title(f"Cooling electricity — {h2} min ahead")
    ax.set_xlabel("test timestep"); ax.set_ylabel("kW"); ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("forecast_eval.png", dpi=115)


if __name__ == "__main__":
    main()
