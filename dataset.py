"""
Training-data generator for the ML forecaster (Milestone 3, extended for M5).

The forecaster's job: given the twin's recent history, predict the hottest
rack temperature and the cooling electricity some minutes into the future.
To learn that, we need lots of examples of "state now -> state later".

Key idea about WHY this is ML and not just physics:
In this project the RC model happens to BE the ground truth, so running the
physics forward would be perfect. But in a real data center you do NOT have
the true physics -- you only have sensor logs. The ML forecaster learns the
dynamics purely from those logs, which is exactly what you'd do in the field.
So here we treat the simulator only as a data source and never let the model
peek at the physics equations.

What this file does:
  1. Runs many short "episodes" with randomized load profiles AND randomized
     CRAC setpoint schedules, for variety.
  2. Turns each episode's time series into supervised samples:
        X = features describing the recent past + the setpoint in effect
        y = target value H steps into the future
  3. Splits episodes into train/test (never mixing an episode across both,
     so there's no time leakage).
"""

from __future__ import annotations
import numpy as np

from config import Config, DEFAULT
from thermal_model import DataCenterThermalModel

# Forecast horizons in MINUTES. dt is 10 s, so 1 min = 6 steps.
HORIZONS_MIN = [5, 15, 30]
# How much recent history to feed the model, in steps (6 steps = 1 min).
N_LAGS = 6
# Targets we forecast.
TARGETS = ["t_rack_max", "cooling_elec_kw"]

# CRAC setpoint randomization (M5). The optimizer will move this knob, so the
# forecaster has to have seen it move.
SETPOINT_MIN_C = 16.0
SETPOINT_MAX_C = 24.0
SETPOINT_HOLD_MIN = (15, 31)      # a setpoint is held this many minutes


def random_load_profile(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """
    A randomized but realistic per-rack load profile, so episodes differ.
    Varies the daily peak location, amplitude, noise, and random spikes.
    """
    n_steps = int(cfg.sim.duration_s / cfg.sim.dt_s)
    n_racks = cfg.rack.n_racks
    t = np.linspace(0, 1, n_steps)

    idle = cfg.rack.idle_power_kw
    span = cfg.rack.max_power_kw - idle

    # Random diurnal shape: random phase + amplitude, occasionally two humps.
    phase = rng.uniform(-0.3, 0.3)
    amp = rng.uniform(0.6, 1.0)
    shape = 0.5 - 0.5 * np.cos(2 * np.pi * (t - 0.1 + phase))
    if rng.random() < 0.5:                      # sometimes a second busy period
        shape += 0.3 * (0.5 - 0.5 * np.cos(4 * np.pi * t))
    shape = np.clip(shape * amp, 0, 1)

    base = idle + span * shape
    load = np.tile(base[:, None], (1, n_racks))
    load += rng.normal(0, 0.5, size=load.shape)

    # A few random load spikes on random racks.
    for _ in range(rng.integers(1, 4)):
        s = rng.integers(0, n_steps - 200)
        dur = rng.integers(60, 200)
        r = rng.integers(0, n_racks)
        load[s:s + dur, r] += rng.uniform(2, 5)

    return np.clip(load, idle * 0.5, cfg.rack.max_power_kw)


def random_setpoint_schedule(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """
    Piecewise-constant CRAC setpoint over one episode.

    Held for 15-30 min, then stepped to a new value. The piecewise structure
    matters: the optimizer changes the setpoint at discrete control intervals,
    so the model needs examples of both the steady-state effect of a setpoint
    LEVEL and the transient response to a setpoint CHANGE. An episode with one
    fixed setpoint would only ever teach the first.
    """
    n_steps = int(cfg.sim.duration_s / cfg.sim.dt_s)
    per_min = int(60 / cfg.sim.dt_s)

    sp = np.empty(n_steps)
    k = 0
    while k < n_steps:
        hold = int(rng.integers(*SETPOINT_HOLD_MIN)) * per_min
        sp[k:k + hold] = rng.uniform(SETPOINT_MIN_C, SETPOINT_MAX_C)
        k += hold
    return sp


def run_episode(cfg: Config, rng: np.random.Generator) -> dict:
    """Step the plant once with a random load AND a random setpoint schedule."""
    model = DataCenterThermalModel(cfg)
    load = random_load_profile(cfg, rng)
    setpoint = random_setpoint_schedule(cfg, rng)

    n = load.shape[0]
    series = {k: np.empty(n) for k in
              ["it_power_kw", "t_air", "t_rack_max", "cooling_elec_kw",
               "q_cooling_kw", "setpoint_c"]}

    for k in range(n):
        obs = model.step(load[k], supply_temp_c=setpoint[k])
        series["it_power_kw"][k] = obs["it_power_kw"]
        series["t_air"][k] = obs["T_air"]
        series["t_rack_max"][k] = obs["T_rack_max"]
        series["cooling_elec_kw"][k] = obs["cooling_elec_kw"]
        series["q_cooling_kw"][k] = obs["q_cooling_kw"]
        series["setpoint_c"][k] = setpoint[k]
    return series


def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Causal rolling mean over the last w steps (no peeking at the future)."""
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = (c[w:] - c[:-w]) / w
    pad = np.full(w - 1, out[0] if len(out) else x[0])
    return np.concatenate([pad, out])


def build_features(series: dict,
                   setpoint_override=None) -> tuple[np.ndarray, list[str]]:
    """
    Turn one episode's time series into a feature matrix (one row per timestep).

    Design principle: feed the model SIGNAL, not NOISE. The future temperature
    is smooth (thermal inertia filters out per-step load jitter), so we hand the
    model smoothed trends rather than raw instantaneous values. Concretely:

      - current temps (already smooth) as anchors
      - the CRAC setpoint in effect, plus the cooling gap it creates
      - SMOOTHED load at two timescales (1 min and 5 min rolling means) so the
        model sees the workload trend, not the sensor jitter
      - a smoothed rate-of-change of rack temp (its 'momentum' / direction)
      - a couple of temperature lags for recent trajectory

    Smoothing here is what removes the jumpy predictions: the noisy raw load and
    raw rate-of-change were tricking the trees into reacting to meaningless
    high-frequency wiggle.

    setpoint_override lets the optimizer ask a counterfactual: "what if I held
    the setpoint at X from here?" Accepts a scalar or a full-length array.
    """
    n = len(series["t_rack_max"])
    cols: dict[str, np.ndarray] = {}

    # Smooth versions of the noisy channels
    load_1m = _rolling_mean(series["it_power_kw"], 6)     # 1 min
    load_5m = _rolling_mean(series["it_power_kw"], 30)    # 5 min
    rack_1m = _rolling_mean(series["t_rack_max"], 6)

    # Current (anchor) state — temps are already smooth
    cols["t_rack_max_now"] = series["t_rack_max"]
    cols["t_air_now"] = series["t_air"]
    cols["cooling_elec_kw_now"] = series["cooling_elec_kw"]

    # --- CRAC setpoint: the control knob (M5) ---
    if setpoint_override is None:
        sp = series.get("setpoint_c", np.full(n, DEFAULT.crac.supply_temp_c))
    else:
        sp = np.broadcast_to(np.asarray(setpoint_override, dtype=float), (n,))
    cols["setpoint_now"] = sp
    # The gap is what physically drives cooling: q_cool = gain x max(0, gap).
    # Handing the model the gap directly beats making a linear model infer it.
    cols["cooling_gap"] = np.maximum(0.0, series["t_air"] - sp)

    # Setpoint in effect at each forecast horizon. Known at inference because
    # the optimizer PICKS it — same status as known near-future load.
    for h_min in HORIZONS_MIN:
        h = h_min * 6
        if setpoint_override is None:
            fut = np.empty(n)
            fut[:n - h] = sp[h:]
            fut[n - h:] = sp[-1]        # tail is trimmed by make_supervised anyway
        else:
            fut = sp                     # optimizer holds its candidate constant
        cols[f"setpoint_at_{h_min}m"] = fut


    # Smoothed workload trend (signal, not jitter)
    cols["load_mean_1m"] = load_1m
    cols["load_mean_5m"] = load_5m
    cols["load_trend"] = load_1m - load_5m               # rising or falling?

    # Temperature momentum: smoothed change over the last minute
    rack_roc = np.zeros(n)
    rack_roc[6:] = rack_1m[6:] - rack_1m[:-6]
    cols["t_rack_roc_1m"] = rack_roc

    # Gap between rack and air (drives how fast the rack can shed heat)
    cols["rack_air_gap"] = series["t_rack_max"] - series["t_air"]

    # A few temperature lags for recent trajectory
    for lag in [6, 18]:                                  # 1 min, 3 min back
        for key in ["t_rack_max", "t_air"]:
            shifted = np.empty(n)
            shifted[:lag] = series[key][0]
            shifted[lag:] = series[key][:-lag]
            cols[f"{key}_lag{lag}"] = shifted

    names = list(cols.keys())
    X = np.column_stack([cols[c] for c in names])
    return X, names


def make_supervised(n_episodes: int = 12, cfg: Config = DEFAULT, seed: int = 0):
    """
    Build the full supervised dataset across many episodes.

    Returns a dict with:
      X            feature matrix
      y            dict target->horizon->target vector
      feature_names
      episode_id   which episode each row came from (for a clean split)
    """
    rng = np.random.default_rng(seed)
    max_h = max(HORIZONS_MIN) * 6                    # steps to trim off the end

    X_parts, epi_parts = [], []
    y_parts = {tgt: {h: [] for h in HORIZONS_MIN} for tgt in TARGETS}
    feature_names = None

    for ep in range(n_episodes):
        series = run_episode(cfg, rng)
        X, names = build_features(series)
        feature_names = names
        n = X.shape[0]
        valid = n - max_h                            # rows that have a full future

        X_parts.append(X[:valid])
        epi_parts.append(np.full(valid, ep))
        for tgt in TARGETS:
            for h_min in HORIZONS_MIN:
                h = h_min * 6
                y_parts[tgt][h_min].append(series[tgt][h:h + valid])

    X_all = np.vstack(X_parts)
    epi_all = np.concatenate(epi_parts)
    y_all = {tgt: {h: np.concatenate(y_parts[tgt][h]) for h in HORIZONS_MIN}
             for tgt in TARGETS}

    return {
        "X": X_all,
        "y": y_all,
        "feature_names": feature_names,
        "episode_id": epi_all,
        "n_episodes": n_episodes,
    }


def train_test_split_by_episode(data: dict, test_frac: float = 0.25):
    """Hold out whole episodes for testing so no timestep leaks across the split."""
    n_ep = data["n_episodes"]
    n_test = max(1, int(round(n_ep * test_frac)))
    test_ep = set(range(n_ep - n_test, n_ep))       # last episodes are the test set
    epi = data["episode_id"]
    is_test = np.array([e in test_ep for e in epi])
    return ~is_test, is_test