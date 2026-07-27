"""
Fault injection (Milestone 4).

Runs the plant while injecting a realistic equipment fault over a time window,
so we can test whether the twin notices. Faults modeled:

  hot_spot          extra heat dumped on one rack (a workload runaway or a
                    blocked airflow path). magnitude = extra kW.
  crac_degradation  the cooler loses capacity (fouled coil, low refrigerant).
                    magnitude = fraction of max cooling lost (0.5 = -50%).
  fan_stuck         the cooler responds weakly to warm air (stuck/failing fan).
                    magnitude = fraction of responsiveness lost.

`simulate()` returns the same time-series shape the forecaster consumes, plus a
boolean mask marking when the fault was active (the ground truth we score
detection against).
"""

from __future__ import annotations
import copy
from dataclasses import dataclass
import numpy as np

from config import Config, DEFAULT
from thermal_model import DataCenterThermalModel, make_load_profile

SERIES_KEYS = ["it_power_kw", "t_air", "t_rack_max", "cooling_elec_kw", "q_cooling_kw"]


@dataclass
class Fault:
    kind: str = "none"          # none | hot_spot | crac_degradation | fan_stuck
    start_min: float = 0.0
    end_min: float = 0.0
    magnitude: float = 0.5
    rack: int = 0


def simulate(cfg: Config = DEFAULT, load: np.ndarray | None = None,
             fault: Fault | None = None, seed: int = 0):
    """Step the plant with an optional injected fault. Returns (series, active)."""
    cfg = copy.deepcopy(cfg)                       # don't mutate the shared config
    if load is None:
        load = make_load_profile(cfg, seed=seed)

    model = DataCenterThermalModel(cfg)
    n = load.shape[0]
    dt = cfg.sim.dt_s
    base_gain = cfg.crac.gain_kw_per_k
    base_cap = cfg.crac.max_cooling_kw

    series = {k: np.empty(n) for k in SERIES_KEYS}
    active = np.zeros(n, dtype=bool)

    for k in range(n):
        t_min = k * dt / 60.0
        on = (fault is not None and fault.kind != "none"
              and fault.start_min <= t_min < fault.end_min)
        active[k] = on

        step_load = load[k].copy()

        # Reset cooler to healthy each step, then degrade if the fault is on.
        model.cfg.crac.max_cooling_kw = base_cap
        model.cfg.crac.gain_kw_per_k = base_gain

        if on:
            if fault.kind == "hot_spot":
                step_load[fault.rack] += fault.magnitude
            elif fault.kind == "crac_degradation":
                model.cfg.crac.max_cooling_kw = base_cap * (1.0 - fault.magnitude)
            elif fault.kind == "fan_stuck":
                model.cfg.crac.gain_kw_per_k = base_gain * (1.0 - fault.magnitude)

        obs = model.step(step_load)
        series["it_power_kw"][k] = obs["it_power_kw"]
        series["t_air"][k] = obs["T_air"]
        series["t_rack_max"][k] = obs["T_rack_max"]
        series["cooling_elec_kw"][k] = obs["cooling_elec_kw"]
        series["q_cooling_kw"][k] = obs["q_cooling_kw"]

    return series, active
