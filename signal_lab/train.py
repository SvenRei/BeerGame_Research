"""signal_lab/train.py -- rollout, update, gate, checkpoints, CSVs. No early stopping.

Recipe (reference-adopted, registered decisions R2-R4):
  rewards     r_i = -(c_i + beta * sum_{j!=i} c_j) / 100
  returns     plain discounted Monte-Carlo (no bootstrap -> immune to critic collapse)
  advantage   G - V, standardized per batch
  update      PPO-clip 0.1, k_epochs 4, every `batch_episodes` episodes, grad-norm 0.2
  optimizers  Adam actor 3e-4 / critic 1e-3 (separate), StepLR x0.5 every 2000 episodes
  entropy     ent_start -> ent_end linearly over `anneal_episodes` (ABSOLUTE; never
              derived from the budget -- the D1/c7 coupling is banned by construction)
  stopping    none: fixed budget; `budget_milestones` snapshots double as V(budget)
  selection   R7: trailing-mean monitor (held-out rho=0.9, seeds 60000+); the low-rho
              gate remains logged as a diagnostic. gate rho in {0.15,0.45,0.75} (not the deployment
              0.9); a monitor-only rho=0.9 trace is logged and never selects

Diagnostics: honest explained variance EV = 1 - Var(G - V)/Var(G) at every update,
with a canary (loud warning if EV < 0.05 after `canary_after` episodes). The D1 critic
collapse would be a red line on the first fig15 here, not invisible for seven rounds.

Run-artifact contract (every consumer reads only this):
  runs/<tag>/{config_resolved.yaml, command.txt, metrics_train.csv, metrics_gate.csv,
              metrics_update.csv, ckpt_best.pt, ckpt_budget<N>.pt, ckpt_final.pt,
              eval/...}
Ends with the terminal marker line "[signal] done." on every clean exit.
"""
import argparse
import copy
import csv
import hashlib
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.beer_game import AGENTS as ENV_AGENTS  # noqa: E402
from env.beer_game import BeerGame, STATE_DIM  # noqa: E402
from signal_lab.agent import Critic, SharedActor, orders_from_s  # noqa: E402
from signal_lab.messages import MessageProvider  # noqa: E402

GATE_RHOS = (0.15, 0.45, 0.75)      # selection regimes -- disjoint from deployment 0.9
MONITOR_RHO = 0.9                   # logged, NEVER selects
GATE_SEED_BASE, MONITOR_SEED_BASE = 50_000, 60_000   # disjoint seed spaces


# ---------------------------------------------------------------------------- config
def load_config(path, overrides):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for kv in overrides or []:
        if "=" not in kv:
            raise ValueError(f"--set expects key=value, got {kv!r}")
        k, v = kv.split("=", 1)
        if k not in cfg:
            raise KeyError(f"unknown config key {k!r} (fail-closed: no silent new keys)")
        cur = cfg[k]
        if isinstance(cur, bool):
            cfg[k] = v.lower() in ("1", "true", "yes")
        elif isinstance(cur, list):
            cfg[k] = [int(x) for x in v.strip("[]() ").split(",") if x.strip() != ""]
        elif cur is None:
            # a null default (e.g. obs_order_clip) carries no type to cast to;
            # interpret the literal via yaml so 12 -> int, 0.5 -> float, null -> None.
            cfg[k] = yaml.safe_load(v)
        else:
            cfg[k] = type(cur)(v)
    return cfg


def make_env(cfg, rho=None):
    return BeerGame({"demand_family": cfg.get("demand_family", "ar1"),
                     "ar1_rho": float(cfg["rho"] if rho is None else rho),
                     "ar1_mu": cfg["ar1_mu"], "ar1_sigma": cfg["ar1_sigma"],
                     "poisson_mu": cfg.get("poisson_mu", 8.0),
                     # P1 / P2 treatment keys -- these MUST reach the env; a previous
                     # whitelist silently dropped them, which would have trained
                     # "garbled" arms on ungarbled observations.
                     "dr_lambda_lo": cfg.get("dr_lambda_lo", 4.0),
                     "dr_lambda_hi": cfg.get("dr_lambda_hi", 24.0),
                     "obs_order_clip": cfg.get("obs_order_clip", None)})


def make_provider(cfg, device="cpu"):
    return MessageProvider(cfg["content"], cfg["topology"], cfg["msg_dim"],
                           cfg={"ar1_mu": cfg["ar1_mu"], "ar1_rho": cfg["rho"],
                                    "demand_family": cfg.get("demand_family", "ar1"),
                                    "dr_lambda_lo": cfg.get("dr_lambda_lo", 4.0),
                                    "dr_lambda_hi": cfg.get("dr_lambda_hi", 24.0)},
                           forecaster_path=cfg.get("forecaster_path") or None,
                           device=device)


# ---------------------------------------------------------------------------- rollout
def play_episode(env, actor, provider, seed, sample=True, max_order=100):
    """One episode. Returns dict of trajectories (numpy) + team cost. Deterministic
    given (seed, weights): torch RNG is seeded here; env RNG in env.reset(seed)."""
    torch.manual_seed(seed)
    obs = env.reset(seed=seed)
    provider.reset()
    h = actor.init_hidden()
    O, MSG, A, LOGP, GS, C = [], [], [], [], [], []
    ORD, DEM, INV, BCK = [], [], [], []
    done = False
    with torch.no_grad():
        while not done:
            o_t = torch.tensor(obs)
            m_sent = actor.message(o_t, h)                       # always computed (T-SYM)
            inc = provider.incoming(env, obs, learned_msgs=m_sent.numpy())
            m_t = torch.tensor(inc)
            logits, _, h = actor.cell(o_t, m_t, h)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample() if sample else logits.argmax(-1)
            O.append(obs); MSG.append(inc); A.append(a.numpy())
            LOGP.append(dist.log_prob(a).numpy())
            GS.append(env.global_state())
            s_vals = actor.grid[a].numpy()
            orders = orders_from_s(s_vals, O[-1], max_order)
            obs, costs, done, info = env.step(orders)
            C.append(costs)
            ORD.append(orders)
            DEM.append(float(info["demand"]))
            INV.append([env.inventory[ag] for ag in ENV_AGENTS])
            BCK.append([env.backlog[ag] for ag in ENV_AGENTS])
    C = np.stack(C)
    return dict(obs=np.stack(O), msg=np.stack(MSG), act=np.stack(A),
                logp=np.stack(LOGP), gstate=np.stack(GS), cost=C,
                orders=np.stack(ORD).astype(float), demand=np.asarray(DEM),
                inv=np.asarray(INV, dtype=float), back=np.asarray(BCK, dtype=float),
                team_cost=float(C.sum()))


def gate_eval(actor, provider, cfg, rhos, seed_base, eps_per_rho):
    costs = []
    for ri, rho in enumerate(rhos):
        env = make_env(cfg, rho=rho)
        for k in range(eps_per_rho):
            costs.append(play_episode(env, actor, provider, seed_base + 1000 * ri + k,
                                      sample=False, max_order=cfg["max_order"])
                         ["team_cost"])
    return float(np.mean(costs))


# ---------------------------------------------------------------------------- update
def mc_returns(rew, gamma):
    G = np.zeros_like(rew)
    run = np.zeros(rew.shape[1], dtype=np.float32)
    for t in range(len(rew) - 1, -1, -1):
        run = rew[t] + gamma * run
        G[t] = run
    return G


def _forward_episode(actor, provider, ep, in_graph_msgs):
    """Re-run the actor over one stored episode (BPTT). For content='learned' the
    messages are recomputed IN-GRAPH so the sender->receiver gradient path (DIAL)
    exists; for every other content they are the stored constants."""
    h = actor.init_hidden()
    if not in_graph_msgs:                       # messages are stored constants: run the
        o = torch.tensor(ep["obs"])             # whole sequence through the GRU at once
        m = torch.tensor(ep["msg"])
        x = torch.relu(actor.fc2(torch.relu(actor.fc1(actor._inp(o, m)))))
        out, _ = actor.gru(x, h)
        return actor.action_head(out)                             # [T, N, bins]
    logits_seq = []                             # learned: sequential, in-graph (DIAL)
    for t in range(len(ep["obs"])):
        o_t = torch.tensor(ep["obs"][t])
        m_t = provider.route_in_graph(actor.message(o_t, h))
        logits, _, h = actor.cell(o_t, m_t, h)
        logits_seq.append(logits)
    return torch.stack(logits_seq)                                # [T, N, bins]


def ppo_update(actor, critic, provider, batch, cfg, opt_a, opt_c, ent_coef):
    beta, gamma = float(cfg["beta"]), float(cfg["gamma"])
    in_graph = cfg["content"] == "learned"
    for ep in batch:
        c = ep["cost"]
        ep["rew"] = -(c + beta * (c.sum(1, keepdims=True) - c)) / 100.0
        ep["G"] = mc_returns(ep["rew"], gamma)
    G = torch.tensor(np.concatenate([ep["G"] for ep in batch]))            # [B*T, N]
    gs = torch.tensor(np.concatenate([ep["gstate"] for ep in batch]))
    acts = torch.tensor(np.concatenate([ep["act"] for ep in batch]))
    logp_old = torch.tensor(np.concatenate([ep["logp"] for ep in batch]))
    with torch.no_grad():
        adv = G - critic(gs)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    stats = {}
    for _ in range(int(cfg["k_epochs"])):
        logits = torch.cat([_forward_episode(actor, provider, ep, in_graph)
                            for ep in batch])
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(acts)
        ratio = torch.exp(logp - logp_old)
        clip = float(cfg["eps_clip"])
        pl = -torch.min(ratio * adv,
                        torch.clamp(ratio, 1 - clip, 1 + clip) * adv).mean()
        ent = dist.entropy().mean()
        V = critic(gs)
        vl = torch.nn.functional.mse_loss(V, G)
        opt_a.zero_grad(); opt_c.zero_grad()
        (pl - ent_coef * ent).backward()
        vl.backward()
        gn = torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg["max_grad_norm"])
        torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg["max_grad_norm"])
        opt_a.step(); opt_c.step()
        with torch.no_grad():
            stats = dict(policy_loss=float(pl), value_loss=float(vl),
                         entropy=float(ent),
                         approx_kl=float((logp_old - logp).mean()),
                         clip_fraction=float(((ratio - 1).abs() > clip).float().mean()),
                         grad_norm=float(gn),
                         action_std=float((torch.sqrt(torch.clamp(
                             (dist.probs * actor.grid**2).sum(-1)
                             - ((dist.probs * actor.grid).sum(-1))**2, min=0.0)))
                             .mean()))    # per-state grid spread, deterministic
    with torch.no_grad():                       # honest EV, post-update
        resid = G - critic(gs)
        varG = float(G.var())
        stats["honest_ev"] = 1.0 - float(resid.var()) / varG if varG > 1e-12 else float("nan")
        stats["state_absmean"] = float(gs.abs().mean())
    return stats


# ---------------------------------------------------------------------------- main
def _csv(path, header):
    f = open(path, "w", newline="")
    w = csv.DictWriter(f, fieldnames=header)
    w.writeheader()
    return f, w


def clone_payload(actor, critic, cfg, ep, gate_cost):
    return {"actor": {k: v.clone() for k, v in actor.state_dict().items()},
            "critic": {k: v.clone() for k, v in critic.state_dict().items()},
            "config": copy.deepcopy(cfg), "episode": int(ep),
            "gate_cost": float(gate_cost), "seed": int(cfg["seed"])}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "conf", "signal.yaml"))
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    tag = cfg["tag"] or f"{cfg['content']}_s{cfg['seed']}"
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run = os.path.join(root, "runs", tag)
    os.makedirs(os.path.join(run, "eval"), exist_ok=True)
    resolved = yaml.safe_dump(cfg, sort_keys=True)
    with open(os.path.join(run, "config_resolved.yaml"), "w") as f:
        f.write(resolved)
    with open(os.path.join(run, "command.txt"), "w") as f:
        f.write("$ " + " ".join(sys.argv) + "\n")
    print(f"[signal] tag={tag} config_sha={hashlib.sha256(resolved.encode()).hexdigest()[:12]}")

    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    actor = SharedActor(cfg["msg_dim"], cfg["hidden"], cfg["act_bins"], cfg["s_max"],
                        msg_scale=cfg.get("msg_scale", 100.0))
    critic = Critic(cfg["hidden"])
    provider = make_provider(cfg)
    opt_a = torch.optim.Adam(actor.parameters(), lr=float(cfg["lr_actor"]))
    opt_c = torch.optim.Adam(critic.parameters(), lr=float(cfg["lr_critic"]))
    step_updates = max(1, int(cfg["lr_step_episodes"]) // max(1, int(cfg["batch_episodes"])))
    sch_a = torch.optim.lr_scheduler.StepLR(opt_a, step_size=step_updates, gamma=0.5)
    sch_c = torch.optim.lr_scheduler.StepLR(opt_c, step_size=step_updates, gamma=0.5)

    env = make_env(cfg)
    train_rng = np.random.default_rng(cfg["seed"] + 12345)   # train seeds: disjoint space
    ftr, wtr = _csv(os.path.join(run, "metrics_train.csv"), ["episode", "team_cost"])
    fga, wga = _csv(os.path.join(run, "metrics_gate.csv"),
                    ["episode", "gate_cost", "best", "best_ep", "monitor_rho09",
                     "honest_ev", "entropy", "action_std", "state_absmean"])
    fup, wup = _csv(os.path.join(run, "metrics_update.csv"),
                    ["episode", "policy_loss", "value_loss", "entropy", "approx_kl",
                     "clip_fraction", "grad_norm", "honest_ev", "state_absmean",
                     "lr_actor", "ent_coef"])

    for k in range(int(cfg["warm_up"])):                     # random play, not stored
        e = make_env(cfg)
        e.reset(seed=int(train_rng.integers(1 << 30)))
        done = False
        while not done:
            _, _, done, _ = e.step(np.random.randint(0, cfg["max_order"] + 1, 4))

    best, best_ep = float("inf"), -1
    crit_hist = []
    best_payload = None
    milestones = sorted(int(m) for m in cfg["budget_milestones"])
    batch, last_stats = [], {}
    total = int(cfg["total_episodes"])
    for ep in range(1, total + 1):
        frac = min(1.0, ep / max(1, int(cfg["anneal_episodes"])))
        ent_coef = float(cfg["ent_start"]) + frac * (float(cfg["ent_end"]) - float(cfg["ent_start"]))
        traj = play_episode(env, actor, provider, int(train_rng.integers(1 << 30)),
                            sample=True, max_order=cfg["max_order"])
        wtr.writerow({"episode": ep, "team_cost": f"{traj['team_cost']:.1f}"})
        batch.append(traj)
        if len(batch) >= int(cfg["batch_episodes"]):
            last_stats = ppo_update(actor, critic, provider, batch, cfg,
                                    opt_a, opt_c, ent_coef)
            batch = []
            sch_a.step(); sch_c.step()
            wup.writerow({"episode": ep,
                          **{k: f"{v:.6g}" for k, v in last_stats.items()
                             if k in ("policy_loss", "value_loss", "entropy",
                                      "approx_kl", "clip_fraction", "grad_norm",
                                      "honest_ev", "state_absmean")},
                          "lr_actor": f"{sch_a.get_last_lr()[0]:.2e}",
                          "ent_coef": f"{ent_coef:.5f}"})
            if (last_stats.get("honest_ev", 1.0) < float(cfg["ev_canary"])
                    and ep >= int(cfg["canary_after"])):
                print(f"[signal] WARNING ep {ep}: honest EV "
                      f"{last_stats['honest_ev']:.4f} < {cfg['ev_canary']} -- the "
                      "critic is not explaining returns (see fig15).", flush=True)
        if ep % int(cfg["gate_every"]) == 0:
            g = gate_eval(actor, provider, cfg, GATE_RHOS, GATE_SEED_BASE,
                          int(cfg["gate_episodes_per_rho"]))
            mon = gate_eval(actor, provider, cfg, (MONITOR_RHO,), MONITOR_SEED_BASE,
                            int(cfg["gate_episodes_per_rho"]))
            mark = ""
            # R7: selection criterion is configurable. `monitor` selects on held-out
            # rho=0.9 seeds (MONITOR_SEED_BASE, disjoint from the eval space) because
            # on D-A2 the monitor reproduced the eval ranking exactly (Spearman 1.0)
            # while the low-rho gate scored 0.2 and ranked the run's WORST checkpoint
            # second-best. `select_smooth` averages the last N criterion readings so a
            # single lucky draw cannot win; N=1 restores single-draw behaviour.
            crit_raw = mon if cfg.get("select_on", "monitor") == "monitor" else g
            crit_hist.append(crit_raw)
            crit = float(np.mean(crit_hist[-max(1, int(cfg.get("select_smooth", 3))):]))
            if crit < best - 1e-9:   # mean over min(N, available) -- never leaves a run
                                     # with no ckpt_best (fail-closed on short runs)
                best, best_ep, mark = crit, ep, "  <-- new best (checkpoint saved)"
                best_payload = clone_payload(actor, critic, cfg, ep, crit)
                torch.save(best_payload, os.path.join(run, "ckpt_best.pt"))
            wga.writerow({"episode": ep, "gate_cost": f"{g:.1f}", "best": f"{best:.1f}",
                          "best_ep": best_ep, "monitor_rho09": f"{mon:.1f}",
                          "honest_ev": f"{last_stats.get('honest_ev', float('nan')):.5f}",
                          "entropy": f"{last_stats.get('entropy', float('nan')):.4f}",
                          "action_std": f"{last_stats.get('action_std', float('nan')):.2f}",
                          "state_absmean": f"{last_stats.get('state_absmean', float('nan')):.2f}"})
            for f in (ftr, fga, fup):
                f.flush()
            _mlab = ("DP" if cfg.get("demand_family") == "dr_poisson" else "0.9")
            print(f"[signal] ep {ep:>6}  gate {g:8.1f}  monitor({_mlab}) {mon:8.1f}  "
                  f"EV {last_stats.get('honest_ev', float('nan')):+.4f}{mark}", flush=True)
        if milestones and ep == milestones[0]:
            m = milestones.pop(0)
            if best_payload is not None:
                torch.save({**best_payload, "budget_episodes": m},
                           os.path.join(run, f"ckpt_budget{m}.pt"))
                print(f"[signal] budget milestone {m}: snapshot of best@ep{best_ep} saved",
                      flush=True)
    torch.save(clone_payload(actor, critic, cfg, total, best),
               os.path.join(run, "ckpt_final.pt"))
    for f in (ftr, fga, fup):
        f.close()
    if best_ep < 0:      # no gate ever fired (short run / gate_every > budget):
        # fail-safe, loudly -- a run must NEVER end without a ckpt_best, or every
        # downstream consumer (eval, sweep idempotency) breaks on a missing file.
        torch.save(clone_payload(actor, critic, cfg, ep, float("nan")),
                   os.path.join(run, "ckpt_best.pt"))
        print("[signal] WARNING: no selection gate fired; ckpt_best = final policy")
    print(f"[signal] best {cfg.get('select_on','monitor')} criterion "
          f"(trailing mean of {cfg.get('select_smooth',3)}) {best:.1f} @ ep {best_ep}")
    print("[signal] done.", flush=True)                      # terminal marker (invariant)


if __name__ == "__main__":
    main()
