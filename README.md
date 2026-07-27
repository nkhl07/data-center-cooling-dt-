# Data Center Cooling Digital Twin

A digital twin of a data center's cooling system. It mirrors thermal state,
forecasts temperatures and energy use, and (in later milestones) optimizes
cooling setpoints to lower PUE.

Status: **Milestones 1–4 complete** — physics plant, streaming pipeline + live
dashboard, an ML forecaster that predicts ahead of real time, and fault
injection + anomaly detection built on the forecaster's residuals.

## What's here

| File | Purpose |
|------|---------|
| `config.py` | All physical parameters (racks, room, CRAC, sim). Tune everything here. |
| `thermal_model.py` | The RC thermal plant + a synthetic load profile generator. |
| `run_sim.py` | Drives the plant, validates the physics, writes CSV + plot. |
| `store.py` | SQLite time-series store for streamed sensor readings. |
| `pipeline.py` | Streaming pipeline: producer → bus → consumer → store. |
| `dashboard.py` | Live Streamlit dashboard reading the store. |
| `dataset.py` | Builds ML training data: varied episodes, features, targets. |
| `forecaster.py` | The forecaster (Ridge per target+horizon) + persistence baseline. |
| `train_forecaster.py` | Trains, evaluates vs baseline, plots, saves the model. |
| `faults.py` | Injects equipment faults (hot spot, CRAC loss, stuck fan) into the plant. |
| `anomaly.py` | Residual-based anomaly detector (robust threshold + debounce). |
| `run_fault_demo.py` | Injects a fault, detects it, reports latency, plots the result. |
| `heat_flow_diagram.svg` | Diagram of the heat-flow paths and equations. |

## Quickstart

Install once:

```bash
pip install -r requirements.txt      # use pip3 / python3 on macOS
```

**Milestone 1 — batch sim + physics validation**

```bash
python run_sim.py
```

**Milestone 2 — live streaming + dashboard** (two terminals)

```bash
python pipeline.py --speed 20         # stream readings into SQLite at 20x
python -m streamlit run dashboard.py  # live dashboard in your browser
```

**Milestone 3 — train the forecaster**

```bash
python train_forecaster.py            # writes forecaster.joblib + forecast_eval.png
```

**Milestone 4 — fault injection + anomaly detection**

```bash
python run_fault_demo.py              # writes fault_detection.png
```

## The model

A lumped-capacitance (RC) network: each rack and the room air are thermal
capacitances linked by thermal resistances. Standard reduced-order approach for
real-time data center thermal work, far cheaper than CFD.

```
IT load --> [racks] --R_rack_air--> [room air] --R_air_out--> outside
                                        |
                                     CRAC cooling  (elec = heat / COP)
```

PUE = total facility power / IT power = (IT + cooling electricity) / IT.

## The forecaster (M3)

Predicts the hottest rack temperature and cooling electricity at 5/15/30 min
horizons. Notable choices:

- Learns purely from streamed data, never the physics equations (as you would
  in the field, where the true physics is unknown).
- Predicts the *change* from the current value; the smooth current reading
  anchors each prediction.
- Uses a regularized **linear** model (Ridge) on smoothed trend features. The
  plant is an RC (near-linear) system, so a linear model both fits better and
  predicts more smoothly than gradient-boosted trees. Match the model to the
  physics.
- Benchmarked against a persistence baseline ("later = now").

Test-set accuracy (held-out episodes): rack-temp MAE ~0.19 °C at 5 min to
~0.79 °C at 30 min, beating the baseline by ~31–36%. Cooling-energy predictions
beat the baseline by ~41–50%.

## Fault detection (M4)

The forecaster only learned normal behavior, so when a fault pushes reality away
from its prediction, the residual (actual − predicted) spikes. The detector
learns the normal residual size with robust stats (median + MAD) and alarms when
the z-score exceeds a threshold for several steps in a row (debounce).

Result: a CRAC losing 45% capacity is caught **1.5 min after onset with zero
false alarms**; stuck-fan and hot-spot faults are caught in 0.2–1.0 min. See
`fault_detection.png`.

## Roadmap

- [x] M1: RC thermal plant + validation
- [x] M2: data pipeline (queue/MQTT) + live dashboard
- [x] M3: ML forecaster (temps + cooling energy)
- [x] M4: fault injection + anomaly detection
- [ ] M5: setpoint optimizer, report PUE before/after
- [ ] M6: Docker + write-up + demo
