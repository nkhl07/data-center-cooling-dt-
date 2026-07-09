"""
Time-series store for the digital twin (Milestone 2).

A thin wrapper over SQLite. In a production twin this would be InfluxDB or
TimescaleDB, but SQLite needs zero setup and is perfect for a portfolio demo:
the pipeline writes readings here, the dashboard reads them back live.

One row = one "sensor snapshot" of the whole room at one instant.
"""

from __future__ import annotations
import sqlite3
import json

SCHEMA = """
CREATE TABLE IF NOT EXISTS readings (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                 REAL NOT NULL,   -- unix wall-clock time the reading was produced
    sim_t_min          REAL NOT NULL,   -- simulated minutes since the run started
    it_power_kw        REAL,
    t_air              REAL,
    t_rack_max         REAL,
    q_cooling_kw       REAL,
    cooling_elec_kw    REAL,
    total_facility_kw  REAL,
    pue                REAL,
    t_racks            TEXT              -- JSON list of per-rack temperatures
);
CREATE INDEX IF NOT EXISTS idx_readings_ts ON readings(ts);
"""

COLUMNS = ["ts", "sim_t_min", "it_power_kw", "t_air", "t_rack_max",
           "q_cooling_kw", "cooling_elec_kw", "total_facility_kw", "pue", "t_racks"]


class TimeSeriesStore:
    def __init__(self, path: str = "dc_twin.db"):
        self.path = str(path)
        with sqlite3.connect(self.path) as c:
            c.executescript(SCHEMA)

    def insert(self, r: dict) -> None:
        """Insert one reading dict. t_racks may be a list; it is JSON-encoded."""
        row = dict(r)
        if isinstance(row.get("t_racks"), (list, tuple)):
            row["t_racks"] = json.dumps(list(row["t_racks"]))
        placeholders = ",".join("?" for _ in COLUMNS)
        with sqlite3.connect(self.path) as c:
            c.execute(
                f"INSERT INTO readings ({','.join(COLUMNS)}) VALUES ({placeholders})",
                tuple(row.get(k) for k in COLUMNS),
            )

    def recent(self, limit: int = 500) -> list[dict]:
        """Return the last `limit` readings in chronological (oldest-first) order."""
        with sqlite3.connect(self.path) as c:
            c.row_factory = sqlite3.Row
            rows = c.execute(
                "SELECT * FROM readings ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(x) for x in reversed(rows)]

    def count(self) -> int:
        with sqlite3.connect(self.path) as c:
            return c.execute("SELECT COUNT(*) FROM readings").fetchone()[0]
