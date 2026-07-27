"""
Residual-based anomaly detection (Milestone 4).

Idea: the forecaster learned NORMAL behavior. At each moment it predicts what
the hottest rack temperature will be a few minutes out. When reality drifts
away from that prediction, something abnormal is happening. The prediction
error (the "residual") is the alarm signal.

Pipeline:
  1. compute_residuals: line up each prediction with what actually happened
     H minutes later; residual = actual - predicted.
  2. ResidualDetector.fit: learn how big residuals normally are, using ROBUST
     statistics (median + MAD) so a few outliers don't inflate the threshold.
  3. flags: convert residuals to z-scores and raise an alarm when they exceed
     the threshold for several steps in a row (debounce), which avoids firing
     on single-sample blips.
"""

from __future__ import annotations
import numpy as np

from dataset import build_features


def compute_residuals(series: dict, forecaster, target: str = "t_rack_max",
                      horizon_min: int = 5):
    """
    Return (residual, aligned_prediction) arrays over the series.

    A prediction made at time t targets time t+H. So the prediction we compare
    against the actual value at index j was made H steps earlier (at j-H).
    """
    X, _ = build_features(series)
    pred_future = forecaster.predict(X, target, horizon_min)   # value expected at t+H
    h = horizon_min * 6
    n = len(series[target])

    aligned = np.full(n, np.nan)
    aligned[h:] = pred_future[: n - h]
    resid = np.full(n, np.nan)
    resid[h:] = series[target][h:] - aligned[h:]
    return resid, aligned


class ResidualDetector:
    def __init__(self, k: float = 5.0, debounce: int = 12):
        self.k = k                 # how many robust std devs counts as anomalous
        self.debounce = debounce   # consecutive steps required to raise an alarm
        self.center = 0.0
        self.scale = 1.0

    def fit(self, normal_resid: np.ndarray) -> "ResidualDetector":
        r = normal_resid[~np.isnan(normal_resid)]
        self.center = float(np.median(r))
        mad = float(np.median(np.abs(r - self.center)))
        self.scale = 1.4826 * mad + 1e-9      # MAD -> std-equivalent for normal data
        return self

    def zscore(self, resid: np.ndarray) -> np.ndarray:
        return (resid - self.center) / self.scale

    def flags(self, resid: np.ndarray):
        """Boolean alarm array + the z-scores. Alarms need `debounce` in a row."""
        z = self.zscore(resid)
        raw = np.abs(z) > self.k
        raw = np.where(np.isnan(z), False, raw)

        out = np.zeros_like(raw, dtype=bool)
        run = 0
        for i, v in enumerate(raw):
            run = run + 1 if v else 0
            if run >= self.debounce:
                out[i - self.debounce + 1: i + 1] = True
        return out, z


def detection_latency_min(active: np.ndarray, flags: np.ndarray, dt_s: float = 10.0):
    """Minutes from fault onset to first alarm (None if never detected)."""
    if not active.any():
        return None
    start = int(np.argmax(active))
    fired = np.where(flags[start:])[0]
    if len(fired) == 0:
        return None
    return fired[0] * dt_s / 60.0
