"""
Live dashboard (Milestone 2) -- Streamlit.

Reads the SQLite store the pipeline is writing and shows the twin's state:
current PUE, load, temperatures, and a per-rack view. Auto-refreshes so it
tracks the stream in real time.

Usage (two terminals):
    1)  python pipeline.py --speed 20
    2)  streamlit run dashboard.py
"""

import json
import pandas as pd
import streamlit as st

from store import TimeSeriesStore
from config import DEFAULT

DB = "dc_twin.db"
WINDOW = 600            # how many recent readings to plot
LIMIT_C = DEFAULT.rack.limit_temp_c

st.set_page_config(page_title="DC Cooling Twin", layout="wide")

# Auto-refresh once per second if the helper is installed; otherwise a button.
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=1000, key="tick")
except Exception:
    st.button("Refresh")

st.title("Data Center Cooling — Live Digital Twin")

store = TimeSeriesStore(DB)
rows = store.recent(WINDOW)

if not rows:
    st.info("No data yet. In another terminal run:  `python pipeline.py --speed 20`")
    st.stop()

df = pd.DataFrame(rows)
latest = df.iloc[-1]

# ---- KPI row -------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("PUE (now)", f"{latest['pue']:.3f}")
c2.metric("IT load", f"{latest['it_power_kw']:.1f} kW")
c3.metric("Room air", f"{latest['t_air']:.1f} °C")
margin = LIMIT_C - latest["t_rack_max"]
c4.metric("Hottest rack", f"{latest['t_rack_max']:.1f} °C",
          f"{margin:+.1f} °C to limit",
          delta_color="inverse")

# ---- temperature chart ---------------------------------------------------
st.subheader("Temperatures")
tdf = df[["sim_t_min", "t_air", "t_rack_max"]].copy()
tdf.columns = ["Time (min)", "Room air", "Hottest rack"]
tdf["Rack limit"] = LIMIT_C
st.line_chart(tdf.set_index("Time (min)"))

# ---- energy chart --------------------------------------------------------
col_a, col_b = st.columns(2)
with col_a:
    st.subheader("PUE")
    st.line_chart(df.set_index("sim_t_min")[["pue"]])
with col_b:
    st.subheader("Cooling electricity (kW)")
    st.line_chart(df.set_index("sim_t_min")[["cooling_elec_kw"]])

# ---- per-rack snapshot ---------------------------------------------------
st.subheader("Per-rack temperature (latest snapshot)")
racks = json.loads(latest["t_racks"]) if isinstance(latest["t_racks"], str) else latest["t_racks"]
rack_df = pd.DataFrame(
    {"Rack": [f"Rack {i}" for i in range(len(racks))], "Temp (°C)": racks}
).set_index("Rack")
st.bar_chart(rack_df)

st.caption(f"{len(df)} readings shown • simulated t = {latest['sim_t_min']:.0f} min")
