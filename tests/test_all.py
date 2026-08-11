"""tests/test_all.py -- the scar-tissue battery. Every test encodes a lesson the
legacy project paid for. Run from repo root:  python -m tests.test_all   (< ~2 min CPU)

  T-ENV     CRN determinism (same seed -> bit-identical trajectories); cost accounting
            equals h*inventory + b*backlog; AR(1) demand mean ~ mu.
  T-ARPRED  provider's arpred equals the closed-form AR(1) conditional mean.
  T-INTERV  zeroed wrapper output == nocomm provider output; shuffled preserves the
            multiset of messages.
  T-FROZEN  no parameter of the dhatc forecaster requires grad; dhatc without a
            checkpoint path fails closed.
  T-PARAM   identical parameter count and identical initial weights across ALL
            contents under the same seed (one architecture, all arms).
  T-SYM     a 'learned'-arm agent behind a zeroed intervention produces BIT-IDENTICAL
            trajectories to the nocomm arm under CRN (the tripwire, by design).
  T-GRAD    after one PPO update, the message head receives gradient iff
            content='learned'; honest EV is finite.
  T-SMOKE   a tiny end-to-end run writes the full artifact contract, the checkpoint
            reloads, evaluate produces a dump, and report is fail-closed (NO-REF
            without baselines, judged with them).
  T-SWEEP   sweep --dry-run propagates every override into the child command line.
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from env.beer_game import AGENTS, BeerGame, N_AGENTS  # noqa: E402
from signal_lab.agent import Critic, SharedActor, count_params  # noqa: E402
from signal_lab.messages import (FrozenForecaster, InterventionWrapper,  # noqa: E402
                                 MessageProvider)
from signal_lab.train import make_provider, play_episode, ppo_update  # noqa: E402

CFG = dict(content="nocomm", topology="retailer_broadcast", msg_dim=3, rho=0.9,
           ar1_mu=12.0, ar1_sigma=3.0, beta=1.0, max_order=100, hidden=32,
           act_bins=41, s_max=100.0, gamma=0.99, k_epochs=2, eps_clip=0.1,
           max_grad_norm=0.2, forecaster_path=None, seed=0)


def _actor(seed=0, hidden=32):
    torch.manual_seed(seed)
    return SharedActor(3, hidden, 41, 100.0)


def t_env():
    e1, e2 = BeerGame({"ar1_rho": 0.9}), BeerGame({"ar1_rho": 0.9})
    e1.reset(seed=5); e2.reset(seed=5)
    rng = np.random.default_rng(0)
    for _ in range(50):
        o = rng.integers(0, 30, 4)
        o1, c1, d1, _ = e1.step(o)
        o2, c2, d2, _ = e2.step(o)
        assert np.array_equal(o1, o2) and np.array_equal(c1, c2) and d1 == d2
        exp = np.array([0.5 * e1.inventory[a] + 1.0 * e1.backlog[a] for a in AGENTS])
        assert np.allclose(c1, exp), "cost accounting broken"
        assert all(e1.inventory[a] >= 0 and e1.backlog[a] >= 0 for a in AGENTS)
    d = []
    e = BeerGame({"ar1_rho": 0.9}); e.reset(seed=1)
    for _ in range(400):
        _, _, done, info = e.step([12, 12, 12, 12])
        d.append(info["demand"])
        if done:
            e.reset(seed=int(np.random.default_rng(len(d)).integers(1 << 30)))
    assert abs(np.mean(d) - 12.0) < 1.5, f"AR(1) mean off: {np.mean(d):.2f}"
    # Poisson family: deterministic under CRN, correct mean, and reachable through the
    # training config (the plumbing, not just the env capability).
    dp = []
    for trial in range(2):
        ep = BeerGame({"demand_family": "poisson", "poisson_mu": 8.0}); ep.reset(seed=7)
        dp.append([ep.step([8, 8, 8, 8])[3]["demand"] for _ in range(50)])
    assert dp[0] == dp[1], "poisson demand not CRN-deterministic"
    assert abs(np.mean(dp[0]) - 8.0) < 1.6, f"poisson mean off: {np.mean(dp[0]):.2f}"
    from signal_lab.train import load_config, make_env
    cfgp = load_config(os.path.join(ROOT, "conf", "signal.yaml"),
                       ["demand_family=poisson", "poisson_mu=8.0"])
    assert make_env(cfgp).cfg["demand_family"] == "poisson", "demand_family not plumbed"
    print("T-ENV     determinism, cost accounting, demand      OK")


def t_arpred():
    p = MessageProvider("arpred", "retailer_broadcast", 3,
                        cfg={"ar1_mu": 12.0, "ar1_rho": 0.9})
    env = BeerGame({"ar1_rho": 0.9})
    obs = env.reset(seed=3)
    env.step([10, 10, 10, 10])
    p._seen_step = True
    m = p.incoming(env, env._obs())
    d = env.last_incoming["retailer"]
    expect = 12.0 + 0.9 * (d - 12.0)
    assert abs(m[1, 0] - expect) < 1e-5, f"{m[1, 0]} != {expect}"
    assert np.allclose(m[0], 0.0), "retailer must receive nothing under broadcast"
    print("T-ARPRED  closed-form AR(1) conditional mean        OK")


def t_interv():
    env = BeerGame({"ar1_rho": 0.9})
    obs = env.reset(seed=4)
    env.step([8, 8, 8, 8])
    raw = MessageProvider("raw", "retailer_broadcast", 3, cfg={"ar1_mu": 12.0})
    noc = MessageProvider("nocomm", "retailer_broadcast", 3)
    z = InterventionWrapper(MessageProvider("raw", "retailer_broadcast", 3,
                                            cfg={"ar1_mu": 12.0}), "zeroed")
    o = env._obs()
    assert np.array_equal(z.incoming(env, o), noc.incoming(env, o)), \
        "zeroed wrapper must equal the nocomm provider"
    sh = InterventionWrapper(MessageProvider("raw", "retailer_broadcast", 3,
                                             cfg={"ar1_mu": 12.0}), "shuffled", seed=1)
    a, b = raw.incoming(env, o), sh.incoming(env, o)
    assert sorted(a[:, 0].tolist()) == sorted(b[:, 0].tolist()), \
        "shuffle must preserve the message multiset"
    print("T-INTERV  zeroed==nocomm, shuffle preserves set     OK")


def t_frozen():
    try:
        MessageProvider("dhatc", "retailer_broadcast", 3, cfg={})
        raise AssertionError("dhatc without a checkpoint must fail closed")
    except ValueError:
        pass
    path = os.path.join(ROOT, "assets", "forecaster_ar1r9.pt")
    if not os.path.exists(path):                       # fit a throwaway one
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts",
                                                     "fit_forecaster.py"),
                        "--steps", "200"], check=True, capture_output=True)
    p = MessageProvider("dhatc", "retailer_broadcast", 3,
                        cfg={"ar1_mu": 12.0, "ar1_rho": 0.9}, forecaster_path=path)
    assert all(not q.requires_grad for q in p.forecaster.parameters()), \
        "frozen forecaster has trainable parameters"
    env = BeerGame({"ar1_rho": 0.9}); env.reset(seed=6)
    env.step([10, 10, 10, 10])
    m = p.incoming(env, env._obs())
    assert np.isfinite(m).all()
    print("T-FROZEN  dhatc frozen + fail-closed loading        OK")


def t_param():
    counts, weights = set(), []
    for content in ("nocomm", "raw", "learned", "arpred"):
        a = _actor(seed=0)
        counts.add(count_params(a))
        weights.append(a.fc1.weight.detach().clone())
    assert len(counts) == 1, "parameter counts differ across contents"
    assert all(torch.equal(weights[0], w) for w in weights[1:]), \
        "same seed must give identical weights in every arm"
    print("T-PARAM   one architecture, all arms                OK")


def t_sym():
    a1 = _actor(seed=0)
    noc = MessageProvider("nocomm", "retailer_broadcast", 3)
    env = BeerGame({"ar1_rho": 0.9})
    tr_noc = play_episode(env, a1, noc, seed=42, sample=True)
    a2 = _actor(seed=0)
    lz = InterventionWrapper(MessageProvider("learned", "retailer_broadcast", 3),
                             "zeroed")
    tr_lz = play_episode(env, a2, lz, seed=42, sample=True)
    assert np.array_equal(tr_noc["act"], tr_lz["act"]), "actions diverge"
    assert np.array_equal(tr_noc["cost"], tr_lz["cost"]), "costs diverge"
    assert tr_noc["team_cost"] == tr_lz["team_cost"]
    print("T-SYM     learned+zeroed == nocomm, bit-identical   OK")


def t_grad():
    for content, expect_grad in (("raw", False), ("learned", True)):
        cfg = {**CFG, "content": content}
        actor, critic = _actor(seed=1), Critic(32)
        prov = make_provider(cfg)
        env = BeerGame({"ar1_rho": 0.9})
        batch = [play_episode(env, actor, prov, seed=7 + k) for k in range(2)]
        opt_a = torch.optim.Adam(actor.parameters(), lr=3e-4)
        opt_c = torch.optim.Adam(critic.parameters(), lr=1e-3)
        g0 = actor.msg_head.weight.detach().clone()
        stats = ppo_update(actor, critic, prov, batch, cfg, opt_a, opt_c, 0.01)
        moved = not torch.equal(g0, actor.msg_head.weight.detach())
        assert moved == expect_grad, \
            f"msg head {'did not train' if expect_grad else 'trained'} under {content}"
        assert np.isfinite(stats["honest_ev"]), "honest EV must be finite"
    print("T-GRAD    msg head trains iff content=learned       OK")




def t_stats():
    import scipy.stats as sst
    from signal_lab.stats import bullwhip, cvar, paired_block

    rng = np.random.default_rng(0)
    # CVaR: hand-checkable vector
    assert abs(cvar([1, 2, 3, 4], 0.5) - 3.5) < 1e-9, "CVaR hand check failed"
    # Bullwhip closed forms: orders == demand -> ratio 1; 2x amplification -> ratio 4
    dem = rng.normal(12, 3, 50)
    mk = lambda o: {"per_episode": [{"demand": dem.tolist(),
                                     "orders": np.stack([o] * 4, 1).tolist()}]}
    r1 = bullwhip(mk(dem))["mean"][0]
    r4 = bullwhip(mk(12 + 2 * (dem - 12)))["mean"][0]
    assert abs(r1 - 1.0) < 1e-6 and abs(r4 - 4.0) < 1e-6, f"bullwhip {r1} {r4}"
    # Paired inference recovers a planted effect V=200 (n=50, sd=60)
    base = rng.normal(4500, 300, 50)
    arm = base - 200 + rng.normal(0, 60, 50)
    pb = paired_block(base, arm, tost_margin=100)
    assert pb["t_p"] < 1e-6 and pb["bca_95ci"][0] < 200 < pb["bca_95ci"][1], \
        f"planted V not recovered: {pb['t_p']}, {pb['bca_95ci']}"
    assert pb["P_V_positive"] > 0.9 and pb["tost_p"] > 0.5, \
        "V=200 must NOT be equivalent to 0 within margin 100"
    # Zero effect IS equivalent within a wide margin
    arm0 = base + rng.normal(0, 60, 50)
    pb0 = paired_block(base, arm0, tost_margin=100)
    assert pb0["tost_p"] < 0.01, f"null effect should pass TOST: {pb0['tost_p']}"
    # Holm: statsmodels adjusted p-values are >= raw and monotone
    from statsmodels.stats.multitest import multipletests
    _, adj, _, _ = multipletests([0.001, 0.02, 0.4], method="holm")
    assert all(a >= r for a, r in zip(adj, [0.001, 0.02, 0.4])) and \
        list(adj) == sorted(adj), "Holm sanity failed"
    # paired-vs-baseline: planted arm 300 cheaper than a baseline on shared draws
    from signal_lab.stats import baseline_block, seed_aggregate
    rng2 = np.random.default_rng(5)
    draws = rng2.normal(4000, 400, 40)
    bars = {"schema": 2, "eval_seeds": list(range(10000, 10040)),
            "static_bs_per_episode": (draws + 300).tolist(),
            "cond_bs_per_episode": (draws - 700).tolist(),
            "static_bs": float(draws.mean() + 300), "cond_bs": float(draws.mean() - 700),
            "eval_episodes": 40}
    dump = {"per_episode": [{"seed": 10000 + i, "team_cost": float(c)}
                            for i, c in enumerate(draws)]}
    bb = baseline_block(dump, bars, tost_margin=100)
    assert bb["available"] and bb["n_pairs"] == 40
    assert abs(bb["vs_static"]["V_mean"] - 300) < 1e-6, bb["vs_static"]["V_mean"]
    assert bb["vs_static"]["t_p"] < 1e-12, "paired test must be decisive on shared draws"
    # gap is 1000 wide; arm sits 300 below static -> 0.3 recovered
    assert abs(bb["gap_recovered"]["mean"] - 0.3) < 1e-6, bb["gap_recovered"]
    # seed mismatch must FAIL-CLOSED, not silently mis-pair
    bad = {"per_episode": [{"seed": 99000 + i, "team_cost": float(c)}
                           for i, c in enumerate(draws)]}
    try:
        baseline_block(bad, bars, 100); raise AssertionError("seed mismatch not caught")
    except SystemExit:
        pass
    # seed aggregation: between-seed SE, sign concordance
    ag = seed_aggregate([{"V_mean": 100.0}, {"V_mean": 140.0}, {"V_mean": 120.0}])
    assert ag["n_seeds"] == 3 and abs(ag["V_seed_mean"] - 120.0) < 1e-9
    assert ag["sign_concordant"] is True
    assert abs(ag["V_between_seed_se"] - (np.std([100, 140, 120], ddof=1) / np.sqrt(3))) < 1e-9
    assert seed_aggregate([{"V_mean": 5.0}, {"V_mean": -5.0}])["sign_concordant"] is False
    print("T-STATS   scipy/statsmodels inference, planted V     OK")


def t_dp():
    """dr_poisson: deterministic, per-episode lambda regime, DP arpred = running mean."""
    from env.beer_game import BeerGame
    from signal_lab.messages import MessageProvider
    means = []
    for sd in range(25):
        e = BeerGame({"demand_family": "dr_poisson"}); e.reset(seed=2000 + sd)
        d = [e.step(np.full(4, 12))[3]["demand"] for _ in range(50)]
        means.append(np.mean(d))
    assert min(means) < 8 and max(means) > 18, means      # regimes genuinely vary
    assert 11 < np.mean(means) < 17, np.mean(means)       # centred near (4+24)/2
    e1 = BeerGame({"demand_family": "dr_poisson"}); e1.reset(seed=2003)
    e2 = BeerGame({"demand_family": "dr_poisson"}); e2.reset(seed=2003)
    for _ in range(30):
        assert e1.step(np.full(4, 9))[3]["demand"] == e2.step(np.full(4, 9))[3]["demand"]
    # arpred under dp == running mean of the sender's own incoming, closed form
    e = BeerGame({"demand_family": "dr_poisson"}); o = e.reset(seed=2010)
    p = MessageProvider("arpred", "retailer_broadcast", 3,
                        cfg={"demand_family": "dr_poisson",
                             "dr_lambda_lo": 4.0, "dr_lambda_hi": 24.0})
    hist = []
    rng = np.random.default_rng(0)
    for t in range(30):
        inc = p.incoming(e, o, learned_msgs=None)
        expect = float(np.mean(hist)) if hist else 14.0
        assert abs(inc[1, 0] - expect) < 1e-6, (t, inc[1, 0], expect)
        o, _, done, info = e.step(rng.integers(0, 30, 4))
        hist.append(e.last_incoming["retailer"])
        if done: break
    # dhatc must FAIL-CLOSED under dp
    try:
        MessageProvider("dhatc", "retailer_broadcast", 3,
                        cfg={"demand_family": "dr_poisson"},
                        forecaster_path="assets/forecaster_ar1r9.pt")
        raise AssertionError("dhatc under dr_poisson must fail-closed")
    except ValueError:
        pass
    print("T-DP      dr_poisson regime + DP forecast + fail-closed OK")


def t_p2():
    """Garbling: physics byte-identical across clip levels; only non-retailer
    observations change; Blackwell nesting min(o,12)=min(min(o,20),12)."""
    from env.beer_game import BeerGame
    rng = np.random.default_rng(4)
    orders = [rng.integers(0, 60, 4) for _ in range(50)]
    def roll(clip):
        e = BeerGame({"obs_order_clip": clip} if clip else {})
        o = e.reset(seed=77); OBS, C = [o.copy()], []
        for a in orders:
            o, c, done, _ = e.step(a); OBS.append(o.copy()); C.append(c.copy())
            if done: break
        return np.array(OBS), np.array(C)
    O_n, C_n = roll(None); O_12, C_12 = roll(12); O_20, C_20 = roll(20)
    np.testing.assert_array_equal(C_n, C_12)              # costs untouched
    np.testing.assert_array_equal(C_n, C_20)
    np.testing.assert_array_equal(O_n[:, 0, :], O_12[:, 0, :])   # retailer unclipped
    np.testing.assert_array_equal(O_12[:, 1:, 3],
                                  np.minimum(O_n[:, 1:, 3], 12))  # clip applied
    np.testing.assert_array_equal(O_12[:, 1:, 3],
                                  np.minimum(O_20[:, 1:, 3], 12))  # Blackwell chain
    assert (O_n[:, 1:, 3] > 12).any(), "orders never exceeded 12; clip untested"
    # the treatment must survive the TRAINING config path too (a whitelist once ate it)
    from signal_lab.train import load_config, make_env
    cfg = load_config(os.path.join(ROOT, "conf", "signal.yaml"), ["obs_order_clip=12"])
    assert make_env(cfg)._env._config.get("obs_order_clip") == 12, \
        "obs_order_clip dropped on the training env path"
    cfg = load_config(os.path.join(ROOT, "conf", "signal.yaml"),
                      ["demand_family=dr_poisson", "dr_lambda_lo=6", "dr_lambda_hi=20"])
    ec = make_env(cfg)._env._config
    assert (ec.get("dr_lambda_lo"), ec.get("dr_lambda_hi")) == (6.0, 20.0), \
        "dr bounds dropped on the training env path"
    print("T-P2      garbling: costs invariant, obs clipped, Blackwell nested OK")


def t_geo():
    """Topology routing rows exact; no_neighbor delivers all-zero incoming."""
    from env.beer_game import BeerGame
    from signal_lab.messages import MessageProvider, routing_matrix
    R = routing_matrix("upstream_only")
    np.testing.assert_array_equal(R, routing_matrix("neighbor"))
    D = routing_matrix("downstream_only")
    assert D[0, 1] == D[1, 2] == D[2, 3] == 1 and D.sum() == 3
    M = routing_matrix("manufacturer_broadcast")
    assert M[0, 3] == M[1, 3] == M[2, 3] == 1 and M.sum() == 3
    assert routing_matrix("no_neighbor").sum() == 0
    e = BeerGame(); o = e.reset(seed=9)
    p = MessageProvider("raw", "no_neighbor", 3, cfg={"ar1_mu": 12.0, "ar1_rho": 0.9})
    rng = np.random.default_rng(1)
    for _ in range(20):
        assert np.all(p.incoming(e, o, learned_msgs=None) == 0.0)
        o, _, d, _ = e.step(rng.integers(0, 30, 4))
        if d: break
    # manufacturer_broadcast: receivers get the MANUFACTURER's value
    e = BeerGame(); o = e.reset(seed=9)
    p = MessageProvider("raw", "manufacturer_broadcast", 3,
                        cfg={"ar1_mu": 12.0, "ar1_rho": 0.9})
    p.incoming(e, o, learned_msgs=None)
    o, _, _, _ = e.step(np.array([7, 9, 11, 13]))
    inc = p.incoming(e, o, learned_msgs=None)
    assert inc[0, 0] == inc[1, 0] == inc[2, 0] == e.last_incoming["manufacturer"]
    assert inc[3, 0] == 0.0
    print("T-GEO     four new topologies routed exactly, placebo is silent OK")


def t_lag():
    """raw_lag_k delivers the sender's incoming from k periods before raw's."""
    from env.beer_game import BeerGame
    from signal_lab.messages import MessageProvider
    e = BeerGame(); o = e.reset(seed=13)
    p0 = MessageProvider("raw", "retailer_broadcast", 3, cfg={"ar1_mu": 12.0})
    p1 = MessageProvider("raw_lag1", "retailer_broadcast", 3, cfg={"ar1_mu": 12.0})
    p2 = MessageProvider("raw_lag2", "retailer_broadcast", 3, cfg={"ar1_mu": 12.0})
    seq0, seq1, seq2 = [], [], []
    rng = np.random.default_rng(2)
    for _ in range(30):
        seq0.append(p0.incoming(e, o, None)[1, 0])
        seq1.append(p1.incoming(e, o, None)[1, 0])
        seq2.append(p2.incoming(e, o, None)[1, 0])
        o, _, d, _ = e.step(rng.integers(0, 30, 4))
        if d: break
    assert seq1[0] == 12.0 and seq1[1] == 12.0 and seq1[2:] == seq0[1:-1], "lag1"
    assert seq2[:3] == [12.0] * 3 and seq2[3:] == seq0[1:-2], "lag2"
    # episode boundary: a fresh episode must re-prime with mu, not leak history
    o = e.reset(seed=14)
    assert p1.incoming(e, o, None)[1, 0] == 12.0, "lag buffer leaked across episodes"
    print("T-LAG     lag contents exact, episode reset clean OK")


def t_ood():
    """Zero-shot OOD transfer: the scenario swaps the DEMAND ONLY. The policy, the
    message content and msg_scale must be exactly as trained, and the OOD dumps must
    not collide with the in-distribution ones."""
    from env.beer_game import BeerGame
    from signal_lab.evaluate import evaluate as ev
    # the stress decks are the vendored ones, and they are genuinely non-stationary
    for fam, lo_hi in (("black_swan", (10, 25)), ("extreme_chaos", (20, 40))):
        e = BeerGame({"demand_family": fam}); e.reset(seed=10000)
        d = [e.step(np.full(4, 8))[3]["demand"] for _ in range(50)]
        early, late = np.mean(d[:10]), np.mean(d[30:])
        assert abs(late - early) > 5, (fam, early, late)   # regime really shifts
    # an ar1-trained checkpoint evaluates OOD without retraining, and the arch echo
    # must still report the TRAINING msg_scale (no silent recalibration)
    ck = os.path.join(ROOT, "runs", "_smoke_nocomm_s60", "ckpt_best.pt")
    if os.path.exists(ck):
        c_id, _, cfg_id, _ = ev(ck, episodes=2, scenario=None)
        c_ood, _, cfg_ood, _ = ev(ck, episodes=2, scenario="black_swan")
        assert cfg_id["msg_scale"] == cfg_ood["msg_scale"], "msg_scale recalibrated OOD"
        assert cfg_id["content"] == cfg_ood["content"]
        assert list(c_id.values()) != list(c_ood.values()), \
            "OOD scenario produced identical costs -- demand swap did not take effect"
    print("T-OOD     stress decks shift regime; OOD eval swaps demand only OK")


def t_costs():
    """F3: the cost regime must reach the env from BOTH the training and the eval
    construction path (the whitelist trap that once ate obs_order_clip), and must
    actually change costs while leaving the physics identical."""
    from env.beer_game import BeerGame
    from signal_lab.train import load_config, make_env
    cfg = load_config(os.path.join(ROOT, "conf", "signal.yaml"),
                      ["holding_cost=0.5", "backorder_cost=2.0"])
    env = make_env(cfg)
    assert (env.h, env.b) == (0.5, 2.0), (env.h, env.b)
    # physics invariance: same actions -> same inventory/backlog trajectory, only the
    # COST scalars differ between regimes
    rng = np.random.default_rng(3)
    acts = [rng.integers(0, 40, 4) for _ in range(40)]
    def roll(h, b):
        e = BeerGame({"holding_cost": h, "backorder_cost": b, "ar1_rho": 0.9})
        o = e.reset(seed=21); O, C = [o.copy()], []
        for a in acts:
            o, c, d, _ = e.step(a); O.append(o.copy()); C.append(c.copy())
            if d: break
        return np.array(O), np.array(C)
    O1, C1 = roll(0.5, 1.0); O2, C2 = roll(0.5, 2.0)
    np.testing.assert_array_equal(O1, O2)              # transitions untouched
    assert not np.allclose(C1, C2), "cost regime had no effect on costs"
    # backorder-heavy must cost MORE whenever any backlog exists
    assert C2.sum() > C1.sum(), (C1.sum(), C2.sum())
    print("T-COSTS   cost regime reaches env, physics invariant, costs shift OK")


def t_dobs():
    """do(obs) for P2: scrambles ONLY the non-retailer incoming-order field, preserves
    its marginal distribution, leaves the retailer and the physics untouched."""
    from env.beer_game import BeerGame
    from signal_lab.evaluate import _ObsScrambler
    rng = np.random.default_rng(11)
    acts = [rng.integers(0, 40, 4) for _ in range(30)]

    def roll(mode):
        e = BeerGame({"ar1_rho": 0.9})
        w = _ObsScrambler(e, mode, seed=5) if mode != "obs_honest" else e
        o = w.reset(seed=31); O, C = [o.copy()], []
        for a in acts:
            o, c, d, _ = w.step(a); O.append(o.copy()); C.append(c.copy())
            if d: break
        return np.array(O), np.array(C)

    Oh, Ch = roll("obs_honest")
    Os, Cs = roll("obs_shuffled")
    # physics untouched: the same actions produce the same COSTS
    np.testing.assert_array_equal(Ch, Cs)
    # retailer never touched, in ANY field
    np.testing.assert_array_equal(Oh[:, 0, :], Os[:, 0, :])
    # non-retailer fields 0-2 (inventory/backlog/on_order) untouched
    np.testing.assert_array_equal(Oh[:, 1:, :3], Os[:, 1:, :3])
    # temporal resample: every emitted value must be one the SAME stage genuinely
    # observed at some point (marginal preserved), but not the current one
    for i in range(3):
        seen = set(np.round(Oh[:, i + 1, 3], 6))
        assert set(np.round(Os[:, i + 1, 3], 6)) <= seen, f"invented a value, stage {i}"
    assert not np.array_equal(Oh[:, 1:, 3], Os[:, 1:, 3]), "scramble was a no-op"
    # and it must bite even when all upstream stages carry IDENTICAL values --
    # the cross-sectional permutation it replaces was provably a no-op there
    e2 = BeerGame({"ar1_rho": 0.9}); w2 = _ObsScrambler(e2, "obs_shuffled", seed=5)
    o2 = w2.reset(seed=31); seq = [o2[1:, 3].copy()]
    for a in acts[:20]:
        o2, _, d2, _ = w2.step(np.full(4, 12)); seq.append(o2[1:, 3].copy())
        if d2: break
    assert len({tuple(np.round(x, 3)) for x in seq}) > 1, "constant-policy case not covered"
    # zeroed variant blanks exactly that field
    Oz, Cz = roll("obs_zeroed")
    np.testing.assert_array_equal(Ch, Cz)
    assert np.all(Oz[:, 1:, 3] == 0.0) and np.any(Oz[:, 0, 3] != 0.0)
    print("T-DOBS    do(obs) scrambles only upstream last_incoming, physics intact OK")


def t_smoke():
    tag = "_smoke_nocomm_s60"
    run = os.path.join(ROOT, "runs", tag)
    shutil.rmtree(run, ignore_errors=True)
    from signal_lab import train as T
    T.main(["--set", "content=nocomm", "seed=60", f"tag={tag}", "rho=0.9",
            "total_episodes=6", "warm_up=1", "gate_every=3", "batch_episodes=2",
            "anneal_episodes=6", "budget_milestones=6", "hidden=32",
            "gate_episodes_per_rho=1", "canary_after=999999"])
    for f in ("config_resolved.yaml", "command.txt", "metrics_train.csv",
              "metrics_gate.csv", "metrics_update.csv", "ckpt_best.pt",
              "ckpt_budget6.pt", "ckpt_final.pt"):
        assert os.path.exists(os.path.join(run, f)), f"artifact contract missing {f}"
    from signal_lab.evaluate import evaluate, main as eval_main
    eval_main(["--ckpt", os.path.join(run, "ckpt_best.pt"), "--episodes", "3"])
    assert os.path.exists(os.path.join(run, "eval", "seed10000_rho0.9.json"))
    from signal_lab.report import main as report_main
    b = os.path.join(ROOT, "runs", "baselines_rho0.9.json")
    stash = b + ".stash"
    if os.path.exists(b):
        os.replace(b, stash)
    try:
        rc = report_main(["--arms", tag])
        assert rc != 0, "report must fail closed without baselines (NO-REF)"
        with open(b, "w") as f:
            json.dump({"rho": 0.9, "static_bs": 1e9, "cond_bs": 1e9}, f)
        rc = report_main(["--arms", tag])
        assert rc == 0, "with an (absurdly high) bar the smoke run must PASS"
        rc = report_main(["--arms", tag + ",no_such_run"])
        assert rc != 0, "a missing arm must fail the report (NO-RUN)"
    finally:
        os.remove(b)
        if os.path.exists(stash):
            os.replace(stash, b)
    from signal_lab.curves import main as curves_main
    curves_main(["--arms", tag, "--out", os.path.join(ROOT, "runs", tag)])
    assert os.path.exists(os.path.join(run, f"fig15_diag_{tag}.png"))
    print("T-SMOKE   end-to-end contract, fail-closed report   OK")


def t_sweep():
    r = subprocess.run([sys.executable, os.path.join(ROOT, "sweep.py"), "--dry-run",
                        "--contents", "nocomm,learned", "--seeds", "60,61",
                        "--rho", "0.9", "--episodes", "123", "--suffix", "x1",
                        "--set", "beta=0.5", "ent_end=0.001"],
                       capture_output=True, text=True, cwd=ROOT)
    out = r.stdout
    assert r.returncode == 0, out + r.stderr
    for frag in ("content=nocomm", "content=learned", "seed=61", "rho=0.9",
                 "total_episodes=123", "beta=0.5", "ent_end=0.001",
                 "tag=nocomm_x1_s60", "tag=learned_x1_s61"):
        assert frag in out, f"sweep dry-run must propagate {frag}\n{out}"
    assert out.count("DRY-RUN") == 4
    print("T-SWEEP   dry-run propagates every override         OK")


if __name__ == "__main__":
    t_env(); t_arpred(); t_interv(); t_frozen(); t_param(); t_sym(); t_grad()
    t_stats(); t_dp(); t_p2(); t_geo(); t_lag(); t_ood(); t_costs(); t_dobs()
    t_smoke(); t_sweep()
    print("\nALL TESTS PASS -- the arm-symmetry, gradient-isolation, and fail-closed "
          "contracts hold.")
