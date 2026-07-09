"""
Reduced-order RC thermal model of a data center room.

This is the "physical plant" (ground truth) for the digital twin. It is a
lumped-capacitance model: each rack and the room air are thermal capacitances
connected by thermal resistances, exactly the reduced-order approach used in
data center thermal management (far cheaper than CFD, runs in milliseconds).

State vector:
    T_rack[i]  temperature of each rack (deg C)
    T_air      temperature of the room air (deg C)

Energy balance per step:
    rack:  C_rack * dT_rack/dt = P_it - (T_rack - T_air) / R_rack_to_air
    air:   C_air  * dT_air/dt  = sum_i (T_rack_i - T_air)/R_rack_to_air
                                 - Q_cooling
                                 + (T_outside - T_air)/R_air_to_outside

Q_cooling is heat removed by the CRAC. Electrical cooling power = Q_cooling / COP,
which is what feeds PUE.
"""

from __future__ import annotations
import numpy as np
from config import Config, DEFAULT


class DataCenterThermalModel:
    def __init__(self, cfg: Config = DEFAULT):
        self.cfg = cfg
        n = cfg.rack.n_racks
        # Capacitances in kJ/K (energy in kJ, power in kW, time in s -> consistent)
        self.C_rack = cfg.rack.thermal_mass_kj_per_k
        self.C_air = cfg.room.air_mass_kj_per_k
        # Initial state: everything at ambient start temp
        self.T_rack = np.full(n, cfg.sim.ambient_start_c, dtype=float)
        self.T_air = float(cfg.sim.ambient_start_c)

    # ---- cooling ----------------------------------------------------------
    def cooling_heat_kw(self, supply_temp_c: float | None = None) -> float:
        """
        Heat (kW) the CRAC removes from the room air right now.

        This is a proportional controller: the warmer the air is above the
        supply setpoint, the harder the cooler works.

          Step 1: pick the setpoint (default from config, or an override the
                  optimizer will supply in a later milestone).
          Step 2: demand = gain x (how far the air is above setpoint).
                  max(0, ...) means: if the air is already at/below setpoint,
                  the cooler does nothing (you can't "add cold").
          Step 3: clamp to the unit's physical ceiling (max_cooling_kw).
        """
        crac = self.cfg.crac
        setpoint = crac.supply_temp_c if supply_temp_c is None else supply_temp_c   # Step 1
        demand = crac.gain_kw_per_k * max(0.0, self.T_air - setpoint)               # Step 2
        return min(demand, crac.max_cooling_kw)                                     # Step 3

    # ---- one integration step --------------------------------------------
    def step(self, it_power_kw: np.ndarray, supply_temp_c: float | None = None) -> dict:
        """
        Advance the plant by one dt.
        it_power_kw: array of IT (heat) power per rack in kW.
        supply_temp_c: optional CRAC setpoint override (the control knob).
        Returns a dict of observable "sensor" readings.
        """
        cfg = self.cfg
        dt = cfg.sim.dt_s               # timestep in seconds
        R_ra = cfg.room.r_rack_to_air   # thermal resistance rack -> air (K/kW)
        R_ao = cfg.room.r_air_to_outside  # thermal resistance air -> outside (K/kW)
        T_out = cfg.sim.outside_temp_c

        it_power_kw = np.asarray(it_power_kw, dtype=float)

        # ============================================================
        # STEP 1: HEAT FLOWS  (Q = dT / R  -- the thermal Ohm's law)
        # Compute every heat flow at the CURRENT temperatures before we
        # move anything. Heat always flows from hot to cold.
        # ============================================================

        # 1a) Each rack sheds heat into the room air. Positive when the rack
        #     is hotter than the air (the normal case).
        q_rack_to_air = (self.T_rack - self.T_air) / R_ra          # kW, one per rack

        # 1b) The CRAC pulls heat out of the air (proportional controller above).
        q_cool = self.cooling_heat_kw(supply_temp_c)               # kW

        # 1c) The building envelope. If it's warmer outside, heat leaks IN
        #     (positive); if cooler outside, heat leaks out (negative).
        q_envelope = (T_out - self.T_air) / R_ao                   # kW

        # ============================================================
        # STEP 2: TEMPERATURE CHANGES  (dT = (Q / C) * dt)
        # Net heat into a mass, divided by its thermal mass C, times the
        # timestep, gives how much its temperature moves this step.
        # Units check: [kW]=[kJ/s] / [kJ/K] * [s] = [K]. Consistent.
        # ============================================================

        # 2a) Rack net heat = what its servers produce MINUS what escaped to air.
        dT_rack = (it_power_kw - q_rack_to_air) / self.C_rack * dt

        # 2b) Air net heat = everything the racks dumped in, MINUS what the CRAC
        #     removed, PLUS/MINUS the envelope leak.
        dT_air = (q_rack_to_air.sum() - q_cool + q_envelope) / self.C_air * dt

        # ============================================================
        # STEP 3: ADVANCE THE STATE  (Euler integration: x += dx)
        # Apply the deltas. dt is small (10 s) so this simple forward-Euler
        # step is stable and accurate enough for this system.
        # ============================================================
        self.T_rack = self.T_rack + dT_rack
        self.T_air = self.T_air + dT_air

        # ============================================================
        # STEP 4: ENERGY BOOKKEEPING FOR PUE
        # Cooling isn't free: moving q_cool of heat costs q_cool/COP of
        # electricity. PUE = total facility power / useful IT power.
        # Always >= 1.0 because cooling is pure overhead on top of IT.
        # ============================================================
        cooling_elec_kw = q_cool / cfg.crac.cop
        it_total_kw = float(it_power_kw.sum())
        total_facility_kw = it_total_kw + cooling_elec_kw
        pue = total_facility_kw / it_total_kw if it_total_kw > 0 else float("nan")

        return {
            "T_rack": self.T_rack.copy(),
            "T_rack_max": float(self.T_rack.max()),
            "T_air": self.T_air,
            "q_cooling_kw": q_cool,
            "cooling_elec_kw": cooling_elec_kw,
            "it_power_kw": it_total_kw,
            "total_facility_kw": total_facility_kw,
            "pue": pue,
        }


def make_load_profile(cfg: Config, seed: int = 0) -> np.ndarray:
    """
    Synthetic but realistic per-rack IT power over the whole sim.
    Daily-style ramp: low at start, busy mid-run, plus noise and a demand spike.
    Returns array shape (n_steps, n_racks) in kW.
    """
    rng = np.random.default_rng(seed)
    n_steps = int(cfg.sim.duration_s / cfg.sim.dt_s)
    n_racks = cfg.rack.n_racks
    t = np.linspace(0, 1, n_steps)

    # Base diurnal-ish shape between idle and max (peaks around 60% through)
    shape = 0.5 - 0.5 * np.cos(2 * np.pi * (t - 0.1))
    shape = np.clip(shape, 0, 1)

    idle = cfg.rack.idle_power_kw
    span = cfg.rack.max_power_kw - idle
    base = idle + span * shape                      # (n_steps,)

    load = np.tile(base[:, None], (1, n_racks))     # broadcast to all racks
    load += rng.normal(0, 0.4, size=load.shape)     # per-rack noise

    # Inject a load spike on rack 0 for a while (stress test the cooling)
    spike_start = int(0.7 * n_steps)
    spike_end = int(0.8 * n_steps)
    load[spike_start:spike_end, 0] += 4.0

    return np.clip(load, idle * 0.5, cfg.rack.max_power_kw)