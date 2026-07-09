"""
Streaming data pipeline (Milestone 2).

This mimics how a real digital twin ingests live sensor data. Three parts,
deliberately decoupled so each could live on a different machine:

    PRODUCER  steps the RC plant in accelerated real time and emits one
              "sensor reading" (JSON) per timestep.
    BUS       carries readings from producer to consumer. Default is an
              in-process queue (zero setup). An optional MQTT transport is
              included to show the real-world pattern -- enable with --mqtt
              if you have a broker (e.g. mosquitto) running.
    CONSUMER  receives readings and writes them to the SQLite store, which
              the dashboard reads live.

Run:
    python pipeline.py --speed 20        # 20x real time, in-process bus
    python pipeline.py --mqtt            # publish/subscribe over MQTT
"""

from __future__ import annotations
import argparse
import json
import os
import queue
import threading
import time

from config import DEFAULT
from thermal_model import DataCenterThermalModel, make_load_profile
from store import TimeSeriesStore

TOPIC = "dc/sensors/room1"
EOS = "dc/control/eos"          # end-of-stream signal


# ---------------------------------------------------------------------------
# Transports (the "bus"). Both expose publish() and a blocking get().
# ---------------------------------------------------------------------------
class InProcessBus:
    """Thread-safe pub/sub via a queue. No external dependencies."""
    def __init__(self):
        self._q: queue.Queue = queue.Queue()

    def publish(self, topic: str, payload: str) -> None:
        self._q.put((topic, payload))

    def get(self, timeout: float = 1.0):
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self):
        pass


class MqttBus:
    """
    Optional real MQTT transport. Requires `paho-mqtt` and a running broker.
    Kept intentionally minimal -- it demonstrates the publish/subscribe
    pattern a production twin would use.
    """
    def __init__(self, host="localhost", port=1883):
        import paho.mqtt.client as mqtt  # imported lazily so it stays optional
        self._q: queue.Queue = queue.Queue()
        self.client = mqtt.Client()
        self.client.on_message = lambda c, u, m: self._q.put(
            (m.topic, m.payload.decode())
        )
        self.client.connect(host, port, 60)
        self.client.subscribe([(TOPIC, 0), (EOS, 0)])
        self.client.loop_start()

    def publish(self, topic: str, payload: str) -> None:
        self.client.publish(topic, payload)

    def get(self, timeout: float = 1.0):
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self):
        self.client.loop_stop()
        self.client.disconnect()


# ---------------------------------------------------------------------------
# Producer / consumer
# ---------------------------------------------------------------------------
def producer(bus, cfg, speed: float, stop_evt: threading.Event) -> None:
    """Step the plant and publish a reading each timestep, in accelerated time."""
    model = DataCenterThermalModel(cfg)
    load = make_load_profile(cfg)
    n_steps = load.shape[0]
    dt = cfg.sim.dt_s

    for k in range(n_steps):
        if stop_evt.is_set():
            break
        obs = model.step(load[k])
        reading = {
            "ts": time.time(),
            "sim_t_min": round(k * dt / 60.0, 3),
            "it_power_kw": round(obs["it_power_kw"], 3),
            "t_air": round(obs["T_air"], 3),
            "t_rack_max": round(obs["T_rack_max"], 3),
            "q_cooling_kw": round(obs["q_cooling_kw"], 3),
            "cooling_elec_kw": round(obs["cooling_elec_kw"], 3),
            "total_facility_kw": round(obs["total_facility_kw"], 3),
            "pue": round(obs["pue"], 4),
            "t_racks": [round(x, 2) for x in obs["T_rack"].tolist()],
        }
        bus.publish(TOPIC, json.dumps(reading))
        # Accelerated real time: one dt of sim compresses to dt/speed wall seconds.
        time.sleep(dt / speed)

    bus.publish(EOS, "end")


def consumer(bus, store: TimeSeriesStore, stop_evt: threading.Event) -> None:
    """Receive readings off the bus and persist them until end-of-stream."""
    while not stop_evt.is_set():
        item = bus.get(timeout=1.0)
        if item is None:
            continue
        topic, payload = item
        if topic == EOS:
            break
        store.insert(json.loads(payload))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run(speed: float = 20.0, db: str = "dc_twin.db", cfg=DEFAULT,
        use_mqtt: bool = False, reset: bool = True) -> TimeSeriesStore:
    if reset and os.path.exists(db):
        os.remove(db)

    store = TimeSeriesStore(db)
    bus = MqttBus() if use_mqtt else InProcessBus()
    stop = threading.Event()

    ct = threading.Thread(target=consumer, args=(bus, store, stop), daemon=True)
    ct.start()
    try:
        producer(bus, cfg, speed, stop)   # producer runs in the main thread
    except KeyboardInterrupt:
        stop.set()
    finally:
        ct.join(timeout=5)
        bus.close()
    return store


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Data center twin streaming pipeline")
    ap.add_argument("--speed", type=float, default=20.0,
                    help="sim-seconds per wall-second (20 = 20x real time)")
    ap.add_argument("--db", default="dc_twin.db")
    ap.add_argument("--mqtt", action="store_true",
                    help="use MQTT transport (needs paho-mqtt + a broker)")
    args = ap.parse_args()

    print(f"Streaming at {args.speed}x into {args.db} "
          f"via {'MQTT' if args.mqtt else 'in-process bus'} ... (Ctrl-C to stop)")
    s = run(speed=args.speed, db=args.db, use_mqtt=args.mqtt)
    print(f"Done. {s.count()} readings stored in {args.db}.")