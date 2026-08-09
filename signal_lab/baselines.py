"""signal_lab/baselines.py -- the yardsticks, recomputed IN-PROJECT.

The legacy constants (AR_StaticBS 4988.6 etc.) belong to the legacy env build; this
module refits both baselines against THIS environment so the bars are self-consistent
with the ported physics -- and prints them next to the legacy numbers as a port
sanity-check (same ballpark expected, not equality).

  StaticBS  per-echelon fixed order-up-to levels S_i; order = clip(S_i - IP_i, ...)
  CondBS    S_{i,t} = a_i + b_i * (dhat_t - mu), dhat_t = mu + rho (d_{t-1} - mu):
            the AR(1)-conditional base-stock frontier (uses the true demand signal at
            every echelon -- privileged information, hence "frontier")

Both are fitted by coordinate descent on FIT seeds and scored on the eval seed space,
then written to runs/baselines_rho<rho>.json. report.py refuses to judge without this
file (fail-closed NO-REF).
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.beer_game import AGENTS, BeerGame, N_AGENTS  # noqa: E402

FIT_SEED_BASE = 70_000          # disjoint from train / gate / monitor / eval spaces
EVAL_SEED_BASE = 10_000         # same seeds the learner is scored on (paired)


def _episode_cost(env, seed, s_fn):
    """Roll one episode under a base-stock rule. s_fn(obs, d_prev) -> S levels [N]."""
    obs = env.reset(seed=seed)
    if hasattr(s_fn, "new_episode"):
        s_fn.new_episode()
    d_prev = None
    total, done = 0.0, False
    while not done:
        S = s_fn(obs, d_prev)
        ip = np.array([BeerGame.inventory_position(obs[i]) for i in range(N_AGENTS)])
        orders = np.clip(np.round(S - ip), 0, env.max_order)
        obs, costs, done, info = env.step(orders)
        d_prev = info["demand"]
        total += float(costs.sum())
    return total


def _score(cfg, seeds, s_fn):
    return float(np.mean(_score_per_episode(cfg, seeds, s_fn)))


def _score_per_episode(cfg, seeds, s_fn):
    """Per-episode team costs, in seed order. Dumped so stats.py can pair a policy
    against the baselines on IDENTICAL demand draws instead of comparing two means."""
    env = BeerGame(cfg)
    return [float(_episode_cost(env, s, s_fn)) for s in seeds]


def _coord_descent(cfg, seeds, S, grids, sweeps):
    """Coordinate descent over per-coordinate candidate grids (in place)."""
    for _ in range(sweeps):
        for i in range(N_AGENTS):
            best_v, best_c = S[i], np.inf
            for v in grids[i]:
                S[i] = v
                c = _score(cfg, seeds, lambda o, d, S=S: S)
                if c < best_c:
                    best_v, best_c = v, c
            S[i] = best_v
    return S


def fit_static(cfg, seeds):
    """Two-stage search. Stage 1: coarse [0,120] step 8 (an earlier [0,60] grid pinned
    the optimum at the boundary -- S*=[~80]*4 lies well outside it). Stage 2: step-2
    refinement +-8 around each stage-1 coordinate."""
    S = _coord_descent(cfg, seeds, np.full(N_AGENTS, 48.0),
                       [list(range(0, 121, 8))] * N_AGENTS, sweeps=3)
    S = _coord_descent(cfg, seeds, S,
                       [list(range(max(0, int(s) - 8), int(s) + 9, 2)) for s in S],
                       sweeps=2)
    return S, _score(cfg, seeds, lambda o, d, S=S: S)


class _DPCond:
    """Conditional base stock for the regime-uncertainty family: S_i = a_i +
    b_i (lam_hat - lam_mid), where lam_hat is the running mean of realized demand this
    episode (the MLE of the episode's unknown Poisson rate). Stateful; the scorer
    calls new_episode() at every reset."""

    def __init__(self, a, b, mid):
        self.a, self.b, self.mid = np.asarray(a, float), np.asarray(b, float), mid
        self.new_episode()

    def new_episode(self):
        self._sum, self._n = 0.0, 0

    def __call__(self, obs, d_prev):
        if d_prev is not None:
            self._sum += float(d_prev)
            self._n += 1
        lam_hat = (self._sum / self._n) if self._n else self.mid
        return self.a + self.b * (lam_hat - self.mid)


def fit_cond_dp(cfg, seeds, S0):
    mid = (float(cfg["dr_lambda_lo"]) + float(cfg["dr_lambda_hi"])) / 2.0
    a_vec, b_vec = S0.astype(float).copy(), np.zeros(N_AGENTS)
    for _ in range(2):
        for i in range(N_AGENTS):
            best, best_c = (a_vec[i], b_vec[i]), np.inf
            for da, bv in itertools.product((-16, -8, -4, 0, 4, 8, 16),
                                            (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)):
                a_try, b_try = a_vec.copy(), b_vec.copy()
                a_try[i], b_try[i] = a_vec[i] + da, bv
                c = _score(cfg, seeds, _DPCond(a_try, b_try, mid))
                if c < best_c:
                    best, best_c = (a_try[i], b_try[i]), c
            a_vec[i], b_vec[i] = best
    return a_vec, b_vec, _score(cfg, seeds, _DPCond(a_vec, b_vec, mid))


def fit_cond(cfg, seeds, S0):
    mu, rho = float(cfg.get("ar1_mu", 12.0)), float(cfg["ar1_rho"])
    a_vec, b_vec = S0.astype(float).copy(), np.zeros(N_AGENTS)

    def s_fn(obs, d_prev, a=a_vec, b=b_vec):
        dhat = mu + rho * ((mu if d_prev is None else d_prev) - mu)
        return a + b * (dhat - mu)

    for _ in range(2):
        for i in range(N_AGENTS):
            best, best_c = (a_vec[i], b_vec[i]), np.inf
            for da, bv in itertools.product((-16, -8, -4, 0, 4, 8, 16),
                                            (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)):
                a_try, b_try = a_vec.copy(), b_vec.copy()
                a_try[i], b_try[i] = a_vec[i] + da, bv
                c = _score(cfg, seeds,
                           lambda o, d, a=a_try, b=b_try:
                           a + b * ((mu + rho * ((mu if d is None else d) - mu)) - mu))
                if c < best_c:
                    best, best_c = (a_try[i], b_try[i]), c
            a_vec[i], b_vec[i] = best
    return a_vec, b_vec, _score(cfg, seeds, s_fn)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, default=0.9)
    ap.add_argument("--demand-family", default="ar1",
                    choices=("ar1", "poisson", "dr_poisson"))
    ap.add_argument("--poisson-mu", type=float, default=8.0)
    ap.add_argument("--fit-episodes", type=int, default=12)
    ap.add_argument("--eval-episodes", type=int, default=50)
    a = ap.parse_args(argv)
    cfg = {"demand_family": a.demand_family, "ar1_rho": a.rho, "ar1_mu": 12.0,
           "ar1_sigma": 3.0, "poisson_mu": a.poisson_mu,
           "dr_lambda_lo": 4.0, "dr_lambda_hi": 24.0}
    fit_seeds = [FIT_SEED_BASE + k for k in range(a.fit_episodes)]
    eval_seeds = [EVAL_SEED_BASE + k for k in range(a.eval_episodes)]

    S, _ = fit_static(cfg, fit_seeds)
    static_ep = _score_per_episode(cfg, eval_seeds, lambda o, d, S=S: S)
    static_cost = float(np.mean(static_ep))
    if a.demand_family == "dr_poisson":
        av, bv, _ = fit_cond_dp(cfg, fit_seeds, S)
        _cond_fn = _DPCond(av, bv, (cfg["dr_lambda_lo"] + cfg["dr_lambda_hi"]) / 2.0)
    else:
        av, bv, _ = fit_cond(cfg, fit_seeds, S)
        mu, rho = float(cfg["ar1_mu"]), a.rho
        _cond_fn = (lambda o, d, av=av, bv=bv:
                    av + bv * ((mu + rho * ((mu if d is None else d) - mu)) - mu))
    cond_ep = _score_per_episode(cfg, eval_seeds, _cond_fn)
    cond_cost = float(np.mean(cond_ep))

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(root, "runs"), exist_ok=True)
    if a.demand_family == "ar1":
        fname = f"baselines_rho{a.rho:g}.json"
    elif a.demand_family == "dr_poisson":
        fname = (f"baselines_dp_{cfg['dr_lambda_lo']:g}-{cfg['dr_lambda_hi']:g}.json")
    else:
        fname = f"baselines_poisson_mu{a.poisson_mu:g}.json"
    out = os.path.join(root, "runs", fname)
    payload = {"demand_family": a.demand_family, "rho": a.rho, "static_bs": static_cost, "cond_bs": cond_cost,
               "static_S": S.tolist(), "cond_a": av.tolist(), "cond_b": bv.tolist(),
               "eval_seed_base": EVAL_SEED_BASE, "eval_episodes": a.eval_episodes,
               "eval_seeds": eval_seeds,
               "static_bs_per_episode": static_ep,
               "cond_bs_per_episode": cond_ep,
               "schema": 2,
               "note": "in-project bars; legacy build reported static 4988.6 / "
                       "frontier 3747.6 at rho=0.9 (ballpark check only)"}
    with open(out, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"[baselines] rho={a.rho:g}  StaticBS {static_cost:.1f} (S={S.tolist()})  "
          f"CondBS {cond_cost:.1f}")
    print(f"[baselines] wrote {out}")


if __name__ == "__main__":
    main()
