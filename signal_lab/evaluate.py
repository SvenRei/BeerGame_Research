"""signal_lab/evaluate.py -- deterministic evaluation of a checkpoint + do(m) probes.

Rebuilds the actor and MessageProvider from the checkpoint's OWN saved config (the
eval-parity lesson: a checkpoint is scored under the exact settings it trained with,
never the current yaml). Writes eval/seed<seed_base>.json into the checkpoint's run
dir per the artifact contract; stdout is never suppressed.

Interventions wrap the provider: --intervention honest|zeroed|shuffled|cross.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.beer_game import BeerGame  # noqa: E402
from signal_lab.agent import Critic, SharedActor  # noqa: E402
from signal_lab.messages import InterventionWrapper, MessageProvider  # noqa: E402
from signal_lab.train import play_episode  # noqa: E402

EVAL_SEED_BASE = 10_000        # disjoint from train / gate / monitor spaces


def load_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    actor = SharedActor(cfg["msg_dim"], cfg["hidden"], cfg["act_bins"], cfg["s_max"])
    actor.load_state_dict(payload["actor"])
    actor.eval()
    critic = Critic(cfg["hidden"])
    critic.load_state_dict(payload["critic"])
    critic.eval()
    provider = MessageProvider(cfg["content"], cfg["topology"], cfg["msg_dim"],
                               cfg={"ar1_mu": cfg["ar1_mu"], "ar1_rho": cfg["rho"]},
                               forecaster_path=cfg.get("forecaster_path") or None)
    return actor, critic, provider, cfg


def evaluate(ckpt_path, episodes=50, rho=None, intervention="honest",
             seed_base=EVAL_SEED_BASE):
    actor, _, provider, cfg = load_checkpoint(ckpt_path)
    if intervention != "honest":
        provider = InterventionWrapper(provider, intervention, seed=seed_base)
    rho = float(cfg["rho"] if rho is None else rho)
    env = BeerGame({"demand_family": cfg.get("demand_family", "ar1"),
                    "poisson_mu": cfg.get("poisson_mu", 8.0),
                    "ar1_rho": rho, "ar1_mu": cfg["ar1_mu"],
                    "ar1_sigma": cfg["ar1_sigma"]})
    costs, recs = {}, []
    for k in range(int(episodes)):
        tr = play_episode(env, actor, provider, seed_base + k,
                          sample=False, max_order=cfg["max_order"])
        costs[str(k)] = tr["team_cost"]
        h, b = float(env.h), float(env.b)
        recs.append({
            "seed": seed_base + k,
            "team_cost": tr["team_cost"],
            "cost_agent": tr["cost"].sum(0).tolist(),
            "cost_hold": (h * tr["inv"]).sum(0).tolist(),
            "cost_back": (b * tr["back"]).sum(0).tolist(),
            "orders": tr["orders"].tolist(),          # [T, N] executed orders
            "demand": tr["demand"].tolist(),          # [T]  realized customer demand
            "back_retailer": tr["back"][:, 0].tolist(),
            "actions": tr["act"].tolist(),            # [T, N] S-bin indices
            "msg0": tr["msg"][:, :, 0].tolist(),      # [T, N] incoming channel slot 0
            "d_prev": tr["obs"][:, 0, 3].tolist(),    # [T]  retailer's observed d_{t-1}
        })
    return costs, recs, cfg, rho


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--rho", type=float, default=None,
                    help="default: the checkpoint's own training rho")
    ap.add_argument("--intervention", default="honest",
                    choices=("honest", "zeroed", "shuffled", "cross"))
    ap.add_argument("--seed-base", type=int, default=EVAL_SEED_BASE, dest="sb")
    a = ap.parse_args(argv)
    costs, recs, cfg, rho = evaluate(a.ckpt, a.episodes, a.rho, a.intervention, a.sb)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(a.ckpt)), "eval")
    os.makedirs(out_dir, exist_ok=True)
    suffix = "" if a.intervention == "honest" else f"_{a.intervention}"
    out = os.path.join(out_dir, f"seed{a.sb}_rho{rho:g}{suffix}.json")
    with open(out, "w") as f:
        json.dump(costs, f)
    traj_out = os.path.join(out_dir, f"seed{a.sb}_rho{rho:g}{suffix}_traj.json")
    with open(traj_out, "w") as f:
        json.dump({"schema": 2, "content": cfg["content"], "rho": rho,
                   "intervention": a.intervention, "seed_base": a.sb,
                   "episodes": int(a.episodes), "per_episode": recs}, f)
    m = float(np.mean(list(costs.values())))
    print(f"[eval] {os.path.basename(os.path.dirname(os.path.abspath(a.ckpt)))}  "
          f"content={cfg['content']} intervention={a.intervention} rho={rho:g}  "
          f"episodes={a.episodes}  mean team cost {m:.1f}")
    print(f"[eval] wrote {out}")
    return m


if __name__ == "__main__":
    main()
