"""scripts/audit_garbling.py -- P2 pre-flight. Costs minutes; can save 90 training jobs.

P2 claims: coarsening the order stream that upstream partners observe RAISES the value
of direct demand sharing, because they can no longer reconstruct demand from orders.

That mechanism has a measurable precondition: the order stream must actually CARRY
recoverable demand information, and clipping must actually DESTROY a meaningful share
of it. If clipping barely dents recoverability, Gamma will be ~0 and P2 fails for a
boring reason (weak manipulation), not an interesting one.

This script measures, on the vendored env with a base-stock driver, how well a
non-retailer can predict current customer demand from its OWN observable history --
with and without obs_order_clip. Reported as out-of-sample R^2 from a linear
autoregression on the observed incoming-order window (a lower bound on what a GRU
could extract, which is the right direction for a go/no-go: if even the linear
predictor loses little, the manipulation is weak).

    python scripts/audit_garbling.py --clips 12 20 --episodes 60

Registered use: run BEFORE the campaign, outcome-blind (it never touches an RL policy).
Decision rule stated in the output.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from env.beer_game import AGENTS, BeerGame  # noqa: E402

AUDIT_SEED_BASE = 95_000          # disjoint from every other seed space
WINDOW = 4                        # lags of observed incoming used as predictors


def collect(clip, episodes, rho, S):
    """Roll base-stock episodes; record each stage's OBSERVED incoming order and the
    true customer demand of that period."""
    cfg = {"ar1_rho": rho}
    if clip is not None:
        cfg["obs_order_clip"] = int(clip)
    env = BeerGame(cfg)
    obs_hist, dem_hist = [], []
    for k in range(episodes):
        o = env.reset(seed=AUDIT_SEED_BASE + k)
        ep_o, ep_d, done = [], [], False
        while not done:
            ip = np.array([BeerGame.inventory_position(o[i]) for i in range(4)])
            o, _, done, info = env.step(np.clip(np.round(S - ip), 0, env.max_order))
            ep_o.append(o[:, 3].copy())          # observed last_incoming per stage
            ep_d.append(info["demand"])
        obs_hist.append(np.array(ep_o)), dem_hist.append(np.array(ep_d))
    return obs_hist, dem_hist


def r2_recover(obs_hist, dem_hist, agent_idx):
    """Out-of-sample R^2 predicting d_t from that agent's own observed incoming window."""
    X, y = [], []
    for O, d in zip(obs_hist, dem_hist):
        for t in range(WINDOW, len(d)):
            X.append(O[t - WINDOW:t, agent_idx])
            y.append(d[t])
    X, y = np.array(X, float), np.array(y, float)
    n = len(y)
    cut = int(0.7 * n)
    Xtr = np.c_[np.ones(cut), X[:cut]]
    Xte = np.c_[np.ones(n - cut), X[cut:]]
    beta, *_ = np.linalg.lstsq(Xtr, y[:cut], rcond=None)
    pred = Xte @ beta
    ss_res = ((y[cut:] - pred) ** 2).sum()
    ss_tot = ((y[cut:] - y[cut:].mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", type=int, default=[12, 20])
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--rho", type=float, default=0.9)
    ap.add_argument("--S", nargs=4, type=float, default=[72, 86, 90, 74],
                    help="base-stock levels of the driver policy (StaticBS fit)")
    a = ap.parse_args()
    S = np.array(a.S, float)

    print(f"P2 GARBLING AUDIT  rho={a.rho:g}  episodes={a.episodes}  window={WINDOW}")
    print("out-of-sample R^2: recovering CURRENT demand from a stage's OWN observed "
          "incoming orders\n")
    rows = {}
    for clip in [None] + list(a.clips):
        O, D = collect(clip, a.episodes, a.rho, S)
        rows[clip] = [r2_recover(O, D, i) for i in range(1, 4)]   # non-retailers only
        lab = "no clip" if clip is None else f"clip {clip}"
        print(f"  {lab:<10} " + "  ".join(
            f"{AGENTS[i+1][:4]} {rows[clip][i]:+.3f}" for i in range(3)))
    base = np.mean(rows[None])
    print()
    for clip in a.clips:
        got = np.mean(rows[clip])
        drop = base - got
        rel = drop / base if base > 1e-9 else float("nan")
        print(f"  clip {clip}: mean R^2 {base:.3f} -> {got:.3f}   "
              f"absolute drop {drop:+.3f}  ({rel:.0%} of recoverable info destroyed)")
    print("\nDECISION RULE (register before running):")
    print("  relative drop >= 0.30  -> manipulation is strong; run P2 as designed.")
    print("  0.10 <= drop < 0.30    -> weak; expect attenuated Gamma, widen n or")
    print("                            tighten the clip, and say so in the paper.")
    print("  drop < 0.10            -> DO NOT RUN P2 at this clip: a null would be")
    print("                            uninformative about the Blackwell mechanism.")


if __name__ == "__main__":
    main()
