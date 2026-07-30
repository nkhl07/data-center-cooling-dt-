"""
Physical configuration for the data center cooling digital twin.

All parameters live here so the whole model is transparent and tunable.
Values are deliberately simple and physically reasonable, not vendor-exact.
Units are SI unless noted.
"""

from dataclasses import dataclass, field


@dataclass
class RackConfig:
    n_racks: int = 4                 # number of server racks in the room
    max_power_kw: float = 15.0       # peak IT power per rack (kW of heat)
    idle_power_kw: float = 3.0       # baseline power when nearly idle
    thermal_mass_kj_per_k: float = 120.0   # rack heat capacity (kJ/K)
    limit_temp_c: float = 32.0       # rack must stay below this (thermal limit)


@dataclass
class RoomConfig:
    air_mass_kj_per_k: float = 800.0     # heat capacity of room air volume (kJ/K)
    # Thermal resistances (K per kW). Lower R = easier heat flow.
    r_rack_to_air: float = 0.35          # rack surface -> room air
    r_air_to_outside: float = 4.0        # room air -> outside (envelope leak)


@dataclass
class CRACConfig:
    """Computer Room Air Conditioner: removes heat from room air."""
    max_cooling_kw: float = 70.0     # max heat it can pull at full tilt
    supply_temp_c: float = 18.0      # cold air supply setpoint
    # Cooling power is proportional to (air_temp - supply_temp), capped at max.
    gain_kw_per_k: float = 8.0       # how aggressively it reacts to warm air
     # Electrical: cooling isn't free. Cooling electricity = heat removed / COP.
    # COP is NOT constant: pumping heat uphill gets cheaper as the lift shrinks,
    # so a warmer supply setpoint buys real efficiency. Carnot would give ~26%/K
    # here; 3%/K is the conservative real-equipment figure (fixed losses don't
    # shrink with the lift). This is the main lever the M5 optimizer exploits.
    cop: float = 4.0                 # COP quoted AT cop_ref_temp_c
    cop_ref_temp_c: float = 18.0     # supply temp the quoted COP applies to
    cop_gain_per_k: float = 0.03     # +3% COP per K of warmer supply air
    cop_min: float = 1.5             # floor, keeps COP sane at extreme setpoints

    def cop_at(self, supply_temp_c: float) -> float:
        """Effective COP at a given supply setpoint."""
        scaled = self.cop * (
            1.0 + self.cop_gain_per_k * (supply_temp_c - self.cop_ref_temp_c)
        )
        return max(self.cop_min, scaled)


@dataclass
class SimConfig:
    dt_s: float = 10.0               # integration timestep (seconds)
    duration_s: float = 6 * 3600.0   # total simulated time (6 hours)
    outside_temp_c: float = 28.0     # ambient outside temperature
    ambient_start_c: float = 22.0    # initial room + rack temperature


@dataclass
class Config:
    rack: RackConfig = field(default_factory=RackConfig)
    room: RoomConfig = field(default_factory=RoomConfig)
    crac: CRACConfig = field(default_factory=CRACConfig)
    sim: SimConfig = field(default_factory=SimConfig)


DEFAULT = Config()
