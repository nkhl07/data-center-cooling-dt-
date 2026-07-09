"""
The ML forecaster (Milestone 3).

Predicts, for several horizons ahead, two things the twin cares about:
  - t_rack_max     the hottest rack temperature  (safety)
  - cooling_elec_kw the cooling electricity        (efficiency)

Design choices, and why:
  * One model per (target, horizon). This "direct multi-horizon" approach is
    simple and robust: each model specializes in predicting exactly H minutes
    out, instead of chaining one-step predictions and accumulating error.
  * Regularized LINEAR model (Ridge) on standardized features. This is a
    deliberate, physics-informed choice: the plant is an RC thermal network,
    whose dynamics are nearly linear. A linear model therefore both fits BETTER
    and predicts far more SMOOTHLY than gradient-boosted trees here (trees
    output piecewise-constant jumps that show up as jitter). If the plant had
    strong nonlinearities -- saturating cooling, on/off equipment, sharp
    thresholds -- trees or an LSTM would earn their keep. Match the model to
    the physics. (See the note at the bottom for the LSTM swap.)
  * Predict the CHANGE from the current value, not the absolute future value,
    so the smooth current reading anchors each prediction.
  * A persistence baseline ("later = now") to prove the ML adds value. If the
    model can't beat "assume nothing changes," it isn't learning.
"""

from __future__ import annotations
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

from dataset import HORIZONS_MIN, TARGETS


def _rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


class TwinForecaster:
    """Holds one regressor per (target, horizon) plus the feature schema."""

    def __init__(self):
        self.models: dict[tuple[str, int], object] = {}
        self.feature_names: list[str] | None = None
        # index of the "current value" feature for each target, used by the
        # persistence baseline (predict = current value).
        self._now_col: dict[str, int] = {}

    # ---- training --------------------------------------------------------
    def fit(self, X, y, feature_names, train_mask):
        self.feature_names = feature_names
        self._now_col = {
            "t_rack_max": feature_names.index("t_rack_max_now"),
            "cooling_elec_kw": feature_names.index("cooling_elec_kw_now"),
        }
        Xtr = X[train_mask]
        for tgt in TARGETS:
            now = Xtr[:, self._now_col[tgt]]
            for h in HORIZONS_MIN:
                # Learn the CHANGE from now, not the absolute future value.
                # The smooth current reading anchors the prediction, so the
                # model only has to predict the (smaller, cleaner) deviation.
                # This both smooths the output and lowers error.
                delta = y[tgt][h][train_mask] - now
                # StandardScaler puts every feature on a common scale so Ridge's
                # single penalty is fair; Ridge's L2 penalty (alpha) keeps the
                # weights small and the model from overfitting.
                model = make_pipeline(StandardScaler(), Ridge(alpha=5.0))
                model.fit(Xtr, delta)
                self.models[(tgt, h)] = model
        return self

    # ---- prediction ------------------------------------------------------
    def predict(self, X, target, horizon_min):
        # prediction = current value + predicted change
        now = X[:, self._now_col[target]]
        return now + self.models[(target, horizon_min)].predict(X)

    def predict_persistence(self, X, target):
        """Baseline: assume the value H minutes from now equals the value now."""
        return X[:, self._now_col[target]]

    # ---- evaluation ------------------------------------------------------
    def evaluate(self, X, y, test_mask) -> dict:
        """Return MAE/RMSE for the model and the baseline, per target+horizon."""
        Xte = X[test_mask]
        report = {}
        for tgt in TARGETS:
            report[tgt] = {}
            for h in HORIZONS_MIN:
                truth = y[tgt][h][test_mask]
                pred = self.predict(Xte, tgt, h)
                base = self.predict_persistence(Xte, tgt)
                report[tgt][h] = {
                    "mae": mean_absolute_error(truth, pred),
                    "rmse": _rmse(truth, pred),
                    "baseline_mae": mean_absolute_error(truth, base),
                    "baseline_rmse": _rmse(truth, base),
                }
        return report

    # ---- persistence to disk --------------------------------------------
    def save(self, path="forecaster.joblib"):
        import joblib
        joblib.dump(
            {"models": self.models,
             "feature_names": self.feature_names,
             "now_col": self._now_col},
            path,
        )

    @classmethod
    def load(cls, path="forecaster.joblib"):
        import joblib
        blob = joblib.load(path)
        obj = cls()
        obj.models = blob["models"]
        obj.feature_names = blob["feature_names"]
        obj._now_col = blob["now_col"]
        return obj


def print_report(report: dict) -> None:
    print("\nForecast accuracy (test set) — lower is better")
    print("=" * 68)
    for tgt, by_h in report.items():
        unit = "°C" if tgt == "t_rack_max" else "kW"
        print(f"\n{tgt}  ({unit})")
        print(f"  {'horizon':>8} | {'model MAE':>10} | {'baseline MAE':>13} | improvement")
        print("  " + "-" * 58)
        for h, m in by_h.items():
            imp = 100 * (1 - m["mae"] / m["baseline_mae"]) if m["baseline_mae"] else 0
            print(f"  {h:>6}m | {m['mae']:>10.3f} | {m['baseline_mae']:>13.3f} | {imp:>6.1f}% better")


# ---------------------------------------------------------------------------
# NOTE — swapping in an LSTM:
# The direct multi-horizon setup makes this easy. Replace HistGradient... with
# a small PyTorch model that consumes the last N_LAGS steps as a sequence and
# outputs one value per horizon. Keep the same fit/predict/evaluate interface
# and everything downstream (optimizer, dashboard) still works unchanged.
# ---------------------------------------------------------------------------
