"""scripts/fit_scales.py -- R9 measurement protocol for message-input divisors.

Registered rule: every message content is delivered to the actor divided by its
stationary standard deviation under the training distribution. This script IS the
protocol: random-order rollouts (uniform integers on [0, 40], the warm-up policy),
seeds SCALE_SEED_BASE+, EPISODES x horizon steps, sd of the incoming slot-0 value at
the first receiver. Deterministic given (content, demand config); computed BEFORE any
training in the cell and never revised afterward. The learned channel is excluded:
its divisor is 100.0 by construction (the message head's tanh range).

Usage:
  python scripts/fit_scales.py --rhos 0,0.3,0.6,0.9            # AR(1) cells
  python scripts/fit_scales.py --dr-poisson 4 24               # DR-Poisson cell
Writes/updates runs/msg_scales.json:  { "<family>|<key>": {content: divisor} }
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from env.beer_game import BeerGame  # noqa: E402
from signal_lab.messages import MessageProvider  # noqa: E402

SCALE_SEED_BASE = 90_000          # disjoint from train/gate/monitor/eval/fit spaces
EPISODES = 8
CONTENTS = ("raw", "ip", "arpred", "dhatc", "raw_lag1", "raw_lag2")


def measure(env_cfg, provider_cfg, forecaster_path):
    out = {}
    for c in CONTENTS:
        try:
            p = MessageProvider(c, "retailer_broadcast", 3, cfg=provider_cfg,
                                forecaster_path=forecaster_path if c == "dhatc" else None)
        except Exception as e:                     # dhatc asset absent for this family
            print(f"  {c:<8} SKIP ({type(e).__name__}: {e})")
            continue
        vals = []
        for s in range(EPISODES):
            env = BeerGame(env_cfg)
            obs = env.reset(seed=SCALE_SEED_BASE + s)
            rng = np.random.default_rng(SCALE_SEED_BASE + s)
            done = False
            while not done:
                inc = p.incoming(env, obs,
                                 learned_msgs=np.zeros((4, 3), dtype=np.float32))
                vals.append(float(inc[1, 0]))
                obs, _, done, _ = env.step(rng.integers(0, 41, 4))
        sd = float(np.std(vals))
        out[c] = round(max(sd, 1e-6), 1)
        print(f"  {c:<8} sd {sd:8.2f}  -> divisor {out[c]}")
    out["learned"] = 100.0
    out["nocomm"] = 100.0                          # inert: channel is identically zero
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rhos", default="0.9")
    ap.add_argument("--ar1-mu", type=float, default=12.0)
    ap.add_argument("--ar1-sigma", type=float, default=3.0)
    ap.add_argument("--dr-poisson", nargs=2, type=float, metavar=("LO", "HI"),
                    default=None)
    ap.add_argument("--forecaster", default=os.path.join(ROOT, "assets",
                                                         "forecaster_ar1r9.pt"))
    a = ap.parse_args()

    path = os.path.join(ROOT, "runs", "msg_scales.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    table = json.load(open(path)) if os.path.exists(path) else {}

    for rho in [float(x) for x in a.rhos.split(",") if x != ""]:
        key = f"ar1|rho{rho:g}"
        print(f"[scales] {key}")
        table[key] = measure({"demand_family": "ar1", "ar1_rho": rho,
                              "ar1_mu": a.ar1_mu, "ar1_sigma": a.ar1_sigma},
                             {"ar1_mu": a.ar1_mu, "ar1_rho": rho,
                              "demand_family": "ar1"}, a.forecaster)
    if a.dr_poisson:
        lo, hi = a.dr_poisson
        key = f"dr_poisson|{lo:g}-{hi:g}"
        print(f"[scales] {key}  (requires demand_family=dr_poisson in the adapter)")
        try:
            table[key] = measure({"demand_family": "dr_poisson",
                                  "dr_lambda_lo": lo, "dr_lambda_hi": hi},
                                 {"demand_family": "dr_poisson",
                                  "dr_lambda_lo": lo, "dr_lambda_hi": hi},
                                 a.forecaster)
        except Exception as e:
            print(f"  FAIL-CLOSED: {type(e).__name__}: {e}")
            print("  (dr_poisson is not yet plumbed through env/beer_game.py -- "
                  "see the capability manifest)")
    with open(path, "w") as f:
        json.dump(table, f, indent=1)
    print(f"[scales] wrote {path}")


if __name__ == "__main__":
    main()
