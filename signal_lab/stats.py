"""signal_lab/stats.py -- the full statistics set over evaluation dumps.

POLICY: every inferential quantity (p-value, interval, correction) is delegated to
scipy.stats or statsmodels; numpy computes descriptives only. Nothing hand-rolled.
Library versions are recorded in the output JSON for the audit trail.

Reads the rich trajectory dumps written by signal_lab/evaluate.py
(runs/<arm>/eval/seed<B>_rho<R>[_<intervention>]_traj.json) and produces, per arm:

  DESCRIPTIVE   mean / sd / se team cost, CVaR_alpha (mean of the worst alpha tail)
  BULLWHIP      Var(orders_i) / Var(customer demand) per echelon, per-episode mean +- se
                (THE Beer Game statistic: order-variance amplification up the chain)
  DECOMPOSITION holding vs backorder cost per echelon
  SERVICE       retailer ready rate (share of periods with zero retailer backlog)
  SIGNALING     Pearson + Spearman dependence of the received channel (slot 0, first
                receiver) on the retailer's observed demand d_{t-1}  [comm arms only]
  LISTENING     causal do(m) contrast on paired seeds: C(zeroed) - C(honest) with
                paired t (scipy.stats.ttest_rel), Wilcoxon, BCa CI; plus the action
                divergence rate honest vs zeroed  [requires a zeroed dump]

and, for each arm against the nocomm reference (paired by identical CRN eval seeds):

  V             per-episode V_k = C_nocomm,k - C_arm,k; mean, se, median, Cohen's d_z,
                P(V>0)
  INFERENCE     paired t-test, Wilcoxon signed-rank, BCa bootstrap CI on mean V
                (scipy.stats.bootstrap, method='BCa'), Schuirmann TOST equivalence
                (statsmodels ttost_paired, symmetric margin --tost-margin), and
                Holm-corrected p-values across the arm family
                (statsmodels multipletests, method='holm')
  GAP           share of the StaticBS -> CondBS optimality gap recovered
                (needs runs/baselines_rho<R>.json)

Fail-closed: a missing dump, a seed mismatch between paired arms, or a NaN anywhere
in the inference inputs aborts with a non-zero exit code. No sentinel values.

Usage:
  python -m signal_lab.stats --nocomm nocomm_s60 --arms raw_s60,dhatc_s60 --rho 0.9
  python -m signal_lab.stats --arms nocomm_s60 --rho 0.9        # descriptives only
"""
import argparse
import json
import os
import sys

import numpy as np
import re

import scipy
import scipy.stats as st
import statsmodels
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.weightstats import ttost_paired

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.beer_game import AGENTS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RE_SEED = re.compile(r"^(.*)_s(\d+)$")


# ------------------------------------------------------------------ loading (fail-closed)
def traj_path(arm, rho, seed_base, intervention="honest"):
    sfx = "" if intervention == "honest" else f"_{intervention}"
    return os.path.join(ROOT, "runs", arm, "eval",
                        f"seed{seed_base}_rho{rho:g}{sfx}_traj.json")


def load_traj(arm, rho, seed_base, intervention="honest"):
    p = traj_path(arm, rho, seed_base, intervention)
    if not os.path.exists(p):
        sys.exit(f"[stats] FAIL-CLOSED: missing dump {p}\n"
                 f"        run: python -m signal_lab.evaluate --ckpt "
                 f"runs/{arm}/ckpt_best.pt --episodes 50"
                 + ("" if intervention == "honest"
                    else f" --intervention {intervention}"))
    with open(p) as f:
        d = json.load(f)
    if d.get("schema") != 2:
        sys.exit(f"[stats] FAIL-CLOSED: {p} has schema {d.get('schema')!r}, need 2 "
                 "(re-run evaluate with the current code)")
    return d


def _costs(d):
    c = np.array([r["team_cost"] for r in d["per_episode"]], dtype=float)
    if not np.all(np.isfinite(c)):
        sys.exit("[stats] FAIL-CLOSED: non-finite team cost in dump")
    return c


def _seeds(d):
    return [r["seed"] for r in d["per_episode"]]


# ------------------------------------------------------------------ descriptive metrics
def cvar(x, alpha):
    """Mean of the worst ceil(alpha*n) episodes (upper tail of cost)."""
    x = np.sort(np.asarray(x, dtype=float))
    k = max(1, int(np.ceil(alpha * len(x))))
    return float(x[-k:].mean())


def bullwhip(d):
    """Per-echelon Var(orders_i)/Var(demand), per episode, then mean +- se.
    Episodes with (near-)constant demand are skipped (ratio undefined)."""
    per_ep = []
    for r in d["per_episode"]:
        dem = np.asarray(r["demand"], dtype=float)
        vd = dem.var(ddof=1)
        if vd < 1e-9:
            continue
        orders = np.asarray(r["orders"], dtype=float)          # [T, N]
        per_ep.append(orders.var(axis=0, ddof=1) / vd)
    if not per_ep:
        return {"mean": [float("nan")] * len(AGENTS), "se": [float("nan")] * len(AGENTS),
                "episodes_used": 0}
    m = np.stack(per_ep)
    return {"mean": m.mean(0).tolist(),
            "se": (m.std(0, ddof=1) / np.sqrt(len(m))).tolist(),
            "episodes_used": int(len(m))}


def decomposition(d):
    hold = np.array([r["cost_hold"] for r in d["per_episode"]], dtype=float).mean(0)
    back = np.array([r["cost_back"] for r in d["per_episode"]], dtype=float).mean(0)
    return {"holding_mean": hold.tolist(), "backorder_mean": back.tolist(),
            "holding_share": float(hold.sum() / max(1e-12, hold.sum() + back.sum()))}


def service(d):
    rr = [float(np.mean(np.asarray(r["back_retailer"]) == 0.0))
          for r in d["per_episode"]]
    mb = [float(np.mean(r["back_retailer"])) for r in d["per_episode"]]
    return {"retailer_ready_rate": float(np.mean(rr)),
            "retailer_mean_backlog": float(np.mean(mb))}


def signaling(d):
    """Dependence of the received channel on the sender's information state.
    Receiver: first non-retailer agent (wholesaler under broadcast/neighbor);
    sender signal: the retailer's observed d_{t-1}. Skips t=0 (mu bootstrap)."""
    if d["content"] == "nocomm":
        return {"applicable": False, "reason": "channel is identically zero"}
    x, y = [], []
    for r in d["per_episode"]:
        msg = np.asarray(r["msg0"], dtype=float)[1:, 1]        # wholesaler, t>=1
        dp = np.asarray(r["d_prev"], dtype=float)[1:]
        x.append(msg); y.append(dp)
    x, y = np.concatenate(x), np.concatenate(y)
    if x.std() < 1e-12:
        return {"applicable": True, "degenerate": True,
                "note": "received channel is constant -- no signaling"}
    pr, pp = st.pearsonr(x, y)
    sr, sp = st.spearmanr(x, y)
    return {"applicable": True, "degenerate": False, "n": int(len(x)),
            "pearson_r": float(pr), "pearson_p": float(pp),
            "spearman_r": float(sr), "spearman_p": float(sp)}


# ------------------------------------------------------------------ paired inference
def _bca_ci(diff, level=0.95, n_resamples=9999, rng=0):
    res = st.bootstrap((np.asarray(diff, dtype=float),), np.mean,
                       confidence_level=level, n_resamples=n_resamples,
                       method="BCa", random_state=np.random.default_rng(rng))
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def paired_block(c_ref, c_arm, tost_margin):
    """Everything about V = c_ref - c_arm, paired per episode. Positive V = arm wins."""
    v = np.asarray(c_ref, dtype=float) - np.asarray(c_arm, dtype=float)
    n = len(v)
    t_stat, t_p = st.ttest_rel(c_ref, c_arm)
    try:
        w_stat, w_p = st.wilcoxon(v)
    except ValueError:                     # all differences zero
        w_stat, w_p = float("nan"), 1.0
    lo, hi = _bca_ci(v)
    tost_p, (tl, tlp, _), (tu, tup, _) = ttost_paired(
        np.asarray(c_ref, dtype=float), np.asarray(c_arm, dtype=float),
        low=-abs(tost_margin), upp=abs(tost_margin))
    sd = v.std(ddof=1)
    return {"n_pairs": int(n),
            "V_mean": float(v.mean()), "V_median": float(np.median(v)),
            "V_se": float(sd / np.sqrt(n)) if n > 1 else float("nan"),
            "cohen_dz": float(v.mean() / sd) if sd > 0 else float("inf"),
            "P_V_positive": float(np.mean(v > 0)),
            "t_stat": float(t_stat), "t_p": float(t_p),
            "wilcoxon_stat": float(w_stat), "wilcoxon_p": float(w_p),
            "bca_95ci": [lo, hi],
            "tost_margin": float(abs(tost_margin)), "tost_p": float(tost_p),
            "tost_lower_p": float(tlp), "tost_upper_p": float(tup)}


def baseline_block(d, bars, tost_margin):
    """Paired comparison against StaticBS and CondBS on IDENTICAL eval seeds.

    A point comparison of two means ("3860.1 < 3870.6") ignores that both are measured
    on the SAME 50 demand draws. Pairing removes the shared demand variance, so the
    same data can support a real interval and p-value on the difference. Also reports
    gap-recovered -- the share of the StaticBS -> CondBS distance captured -- with a
    BCa interval, since a point estimate of a ratio is not defensible on its own."""
    if bars.get("schema") != 2:
        return {"available": False,
                "reason": "baselines file predates per-episode dumps -- re-run "
                          "signal_lab.baselines to enable paired baseline tests"}
    seeds_arm = _seeds(d)
    seeds_bar = bars.get("eval_seeds", [])
    n = min(len(seeds_arm), len(seeds_bar))
    if seeds_arm[:n] != seeds_bar[:n]:
        sys.exit("[stats] FAIL-CLOSED: arm and baselines were evaluated on different "
                 "seeds -- pairing invalid")
    c = _costs(d)[:n]
    st = np.asarray(bars["static_bs_per_episode"][:n], dtype=float)
    cd = np.asarray(bars["cond_bs_per_episode"][:n], dtype=float)
    out = {"available": True, "n_pairs": int(n)}
    for name, ref in (("vs_static", st), ("vs_cond", cd)):
        out[name] = paired_block(ref, c, tost_margin)      # V = baseline - arm
    gap = st - cd                                          # per-episode gap width
    if np.abs(gap.mean()) > 1e-9:
        recov = (st - c) / gap.mean()
        lo, hi = _bca_ci(recov)
        out["gap_recovered"] = {"mean": float(recov.mean()),
                                "se": float(recov.std(ddof=1) / np.sqrt(n)),
                                "bca_95ci": [lo, hi]}
    return out


def seed_aggregate(per_seed):
    """Aggregate V across TRAINING-seed replicates.

    The eval seed space is fixed (10000+), so replicates share demand draws and their
    V estimates are correlated through them. The seed-level mean is therefore reported
    with a between-seed SE over the replicate means -- the quantity that answers "would
    another training seed reproduce this?" -- rather than pooling episodes, which would
    understate uncertainty by treating shared draws as independent."""
    v = np.asarray([p["V_mean"] for p in per_seed], dtype=float)
    k = len(v)
    out = {"n_seeds": int(k), "V_seed_mean": float(v.mean()),
           "V_per_seed": v.tolist(),
           "sign_concordant": bool(np.all(v > 0) or np.all(v < 0))}
    if k >= 2:
        sd = v.std(ddof=1)
        out["V_between_seed_se"] = float(sd / np.sqrt(k))
        t, p = st.ttest_1samp(v, 0.0)
        out["t_stat"], out["t_p"] = float(t), float(p)
        if k >= 3:
            lo, hi = _bca_ci(v, n_resamples=4999)
            out["bca_95ci"] = [lo, hi]
    return out


def listening_block(d_honest, d_zeroed):
    """Causal do(m) contrast: C(zeroed) - C(honest) on paired seeds, plus the
    action divergence rate (share of (t, agent) cells where the chosen S differs)."""
    if _seeds(d_honest) != _seeds(d_zeroed):
        sys.exit("[stats] FAIL-CLOSED: honest and zeroed dumps use different seeds")
    ch, cz = _costs(d_honest), _costs(d_zeroed)
    delta = cz - ch
    div = []
    for rh, rz in zip(d_honest["per_episode"], d_zeroed["per_episode"]):
        ah, az = np.asarray(rh["actions"]), np.asarray(rz["actions"])
        div.append(float((ah != az).mean()))
    if np.allclose(delta, 0.0):
        return {"n_pairs": int(len(delta)), "delta_cost_mean": 0.0,
                "delta_cost_se": 0.0, "t_p": None, "wilcoxon_p": None,
                "bca_95ci": [0.0, 0.0],
                "action_divergence_rate": float(np.mean(div)),
                "degenerate": True,
                "note": "policy identical under do(zeroed) -- no positive listening"}
    t_stat, t_p = st.ttest_rel(cz, ch)
    try:
        _, w_p = st.wilcoxon(delta)
    except ValueError:
        w_p = 1.0
    lo, hi = _bca_ci(delta)
    return {"n_pairs": int(len(delta)),
            "delta_cost_mean": float(delta.mean()),
            "delta_cost_se": float(delta.std(ddof=1) / np.sqrt(len(delta))),
            "t_p": float(t_p), "wilcoxon_p": float(w_p), "bca_95ci": [lo, hi],
            "action_divergence_rate": float(np.mean(div)), "degenerate": False}


# ------------------------------------------------------------------ per-arm assembly
def arm_block(d, cvar_alpha):
    c = _costs(d)
    return {"content": d["content"], "episodes": int(len(c)),
            "cost_mean": float(c.mean()),
            "cost_sd": float(c.std(ddof=1)),
            "cost_se": float(c.std(ddof=1) / np.sqrt(len(c))),
            "cvar": {"alpha": cvar_alpha, "value": cvar(c, cvar_alpha)},
            "bullwhip": bullwhip(d),
            "decomposition": decomposition(d),
            "service": service(d),
            "signaling": signaling(d)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True,
                    help="comma-separated run tags, e.g. raw_s60,dhatc_s60")
    ap.add_argument("--nocomm", default=None,
                    help="nocomm reference tag, or a comma-separated list. With a list, "
                         "each arm is paired against the reference sharing its _s<NN> "
                         "training-seed suffix -- pairing across seeds would fold "
                         "training-seed variance into V.")
    ap.add_argument("--rho", type=float, default=0.9)
    ap.add_argument("--seed-base", type=int, default=10_000, dest="sb")
    ap.add_argument("--tost-margin", type=float, default=250.0,
                    help="symmetric equivalence margin for Schuirmann TOST (cost units)")
    ap.add_argument("--cvar-alpha", type=float, default=0.25)
    ap.add_argument("--label", default="stats")
    a = ap.parse_args(argv)

    arms = [s for s in a.arms.split(",") if s]
    out = {"rho": a.rho, "seed_base": a.sb, "tost_margin": a.tost_margin,
           "libraries": {"numpy": np.__version__, "scipy": scipy.__version__,
                         "statsmodels": statsmodels.__version__},
           "arms": {}, "paired_vs_nocomm": {}, "listening": {}}

    _bpath = os.path.join(ROOT, "runs", f"baselines_rho{a.rho:g}.json")
    _BARS = json.load(open(_bpath)) if os.path.exists(_bpath) else None

    def _seed_suffix(tag):
        m = _RE_SEED.match(tag)
        return m.group(2) if m else None

    refs = {}                      # seed suffix (or None) -> (tag, traj)
    if a.nocomm:
        for tag in [t for t in a.nocomm.split(",") if t]:
            tr = load_traj(tag, a.rho, a.sb)
            refs[_seed_suffix(tag)] = (tag, tr)
            out["arms"][tag] = arm_block(tr, a.cvar_alpha)
            if _BARS is not None:
                out["arms"][tag]["vs_baselines"] = baseline_block(
                    tr, _BARS, a.tost_margin)
        out["nocomm_refs"] = {k: v[0] for k, v in refs.items()}

    def _ref_for(arm):
        """Reference sharing this arm's training seed; fall back to a lone reference."""
        sfx = _seed_suffix(arm)
        if sfx in refs:
            return refs[sfx]
        if len(refs) == 1:
            return next(iter(refs.values()))
        sys.exit(f"[stats] FAIL-CLOSED: no nocomm reference for training seed "
                 f"{sfx!r} (arm {arm}). References present: "
                 f"{sorted(k for k in refs)}")

    print(f"== SIGNAL statistics  rho={a.rho:g}  seed_base={a.sb}  "
          f"scipy {scipy.__version__} / statsmodels {statsmodels.__version__} ==")
    hdr = (f"{'arm':<22}{'mean':>9}{'se':>8}{'CVaR':>9}"
           f"{'bw R/W/D/M':>28}{'ready':>7}")
    print(hdr)
    raw_p = []
    for arm in arms:
        d = load_traj(arm, a.rho, a.sb)
        blk = arm_block(d, a.cvar_alpha)
        out["arms"][arm] = blk
        bw = "/".join(f"{x:.1f}" for x in blk["bullwhip"]["mean"])
        print(f"{arm:<22}{blk['cost_mean']:>9.1f}{blk['cost_se']:>8.1f}"
              f"{blk['cvar']['value']:>9.1f}{bw:>28}"
              f"{blk['service']['retailer_ready_rate']:>7.2f}")
        if refs and arm not in out.get("nocomm_refs", {}).values():
            ref_tag, ref_tr = _ref_for(arm)
            if _seeds(ref_tr) != _seeds(d):
                sys.exit(f"[stats] FAIL-CLOSED: {arm} and {ref_tag} were evaluated "
                         "on different eval seeds -- pairing invalid")
            pb = paired_block(_costs(ref_tr), _costs(d), a.tost_margin)
            pb["paired_against"] = ref_tag
            out["paired_vs_nocomm"][arm] = pb
            raw_p.append((arm, pb["t_p"]))
        if _BARS is not None:
            out["arms"][arm]["vs_baselines"] = baseline_block(d, _BARS, a.tost_margin)
        if out["arms"][arm]["content"] != "nocomm":
            for iv in ("zeroed", "shuffled"):
                if os.path.exists(traj_path(arm, a.rho, a.sb, iv)):
                    out.setdefault("listening", {}).setdefault(arm, {})[iv] = \
                        listening_block(d, load_traj(arm, a.rho, a.sb, iv))

    if raw_p:
        rej, adj, _, _ = multipletests([p for _, p in raw_p], alpha=0.05,
                                       method="holm")
        print(f"\n{'arm':<18}{'vs':<16}{'V mean':>9}{'se':>8}{'dz':>7}{'P(V>0)':>8}"
              f"{'t_p':>10}{'holm_p':>10}{'BCa 95% CI':>22}")
        for (arm, _), hp, rj in zip(raw_p, adj, rej):
            pb = out["paired_vs_nocomm"][arm]
            pb["holm_p"] = float(hp)
            pb["holm_reject_at_05"] = bool(rj)
            ci = pb["bca_95ci"]
            print(f"{arm:<18}{pb.get('paired_against', ''):<16}"
                  f"{pb['V_mean']:>9.1f}{pb['V_se']:>8.1f}"
                  f"{pb['cohen_dz']:>7.2f}{pb['P_V_positive']:>8.2f}"
                  f"{pb['t_p']:>10.2e}{hp:>10.2e}"
                  f"{'[' + format(ci[0], '.1f') + ', ' + format(ci[1], '.1f') + ']':>22}")

    if out["listening"]:
        print(f"\n== do(m) listening: cost INCREASE when the channel is corrupted ==")
        print("   shuffled is the informative control (keeps the message's marginal")
        print("   distribution, destroys only its correlation with the state);")
        print("   zeroed also injects a value never seen in training, so any excess")
        print("   of zeroed over shuffled is distribution shift, not information.")
        print(f"{'arm':<16}{'intervention':<12}{'delta':>9}{'se':>8}{'t_p':>10}"
              f"{'BCa 95% CI':>20}{'act.div':>9}")
        for arm, ivs in out["listening"].items():
            for iv, lb in ivs.items():
                if lb.get("degenerate"):
                    print(f"{arm:<16}{iv:<12}{'0.0':>9}{'--':>8}{'--':>10}"
                          f"{'DEGENERATE':>20}{lb['action_divergence_rate']:>9.1%}")
                    continue
                ci = lb["bca_95ci"]
                print(f"{arm:<16}{iv:<12}{lb['delta_cost_mean']:>+9.1f}"
                      f"{lb['delta_cost_se']:>8.1f}{lb['t_p']:>10.2e}"
                      f"{'[' + format(ci[0], '.0f') + ', ' + format(ci[1], '.0f') + ']':>20}"
                      f"{lb['action_divergence_rate']:>9.1%}")
        fams = {}
        for arm, ivs in out["listening"].items():
            m = _RE_SEED.match(arm)
            for iv, lb in ivs.items():
                fams.setdefault((m.group(1) if m else arm, iv), []).append(
                    lb["delta_cost_mean"])
        print(f"\n{'family':<16}{'intervention':<12}{'mean delta':>12}"
              f"{'seeds':>7}{'sign concordant':>17}")
        out["listening_aggregate"] = {}
        for (fam, iv), vals in fams.items():
            conc = bool(all(v > 0 for v in vals) or all(v < 0 for v in vals))
            out["listening_aggregate"][f"{fam}|{iv}"] = {
                "mean_delta": float(np.mean(vals)), "n_seeds": len(vals),
                "per_seed": vals, "sign_concordant": conc}
            print(f"{fam:<16}{iv:<12}{np.mean(vals):>+12.1f}{len(vals):>7}"
                  f"{str(conc):>17}")
    for arm in arms:
        sg = out["arms"][arm]["signaling"]
        if sg.get("applicable") and not sg.get("degenerate"):
            print(f"[signaling] {arm}: channel vs d_prev  pearson r "
                  f"{sg['pearson_r']:+.3f} (p {sg['pearson_p']:.2e}), spearman "
                  f"{sg['spearman_r']:+.3f}")

    if _BARS is not None and _BARS.get("schema") == 2:
        print(f"\n== paired vs baselines (same {_BARS['eval_episodes']} demand draws) "
              f"StaticBS {_BARS['static_bs']:.1f} -> CondBS {_BARS['cond_bs']:.1f} ==")
        print(f"{'arm':<22}{'V vs Static':>12}{'se':>8}{'t_p':>10}"
              f"{'BCa 95% CI':>22}{'gap recov':>11}{'gap 95% CI':>20}")
        for arm in out["arms"]:
            vb = out["arms"][arm].get("vs_baselines", {})
            if not vb.get("available"):
                continue
            ps, gr = vb["vs_static"], vb.get("gap_recovered", {})
            ci = ps["bca_95ci"]
            gci = gr.get("bca_95ci", [float("nan")] * 2)
            print(f"{arm:<22}{ps['V_mean']:>12.1f}{ps['V_se']:>8.1f}{ps['t_p']:>10.2e}"
                  f"{'[' + format(ci[0], '.0f') + ', ' + format(ci[1], '.0f') + ']':>22}"
                  f"{gr.get('mean', float('nan')):>11.3f}"
                  f"{'[' + format(gci[0], '.2f') + ', ' + format(gci[1], '.2f') + ']':>20}")
        print("  (V vs Static > 0 means the arm is CHEAPER than StaticBS)")
    elif _BARS is not None:
        print("\n[stats] baselines file predates per-episode dumps -- re-run "
              "signal_lab.baselines for paired baseline tests")
    else:
        print("\n[stats] note: baselines file missing -- run signal_lab.baselines")

    # seed-level aggregation across training replicates (tags ending _s<NN>)
    if out["paired_vs_nocomm"]:
        groups = {}
        for arm, pb in out["paired_vs_nocomm"].items():
            m = _RE_SEED.match(arm)
            groups.setdefault(m.group(1) if m else arm, []).append(pb)
        multi = {k: v for k, v in groups.items() if len(v) >= 2}
        if multi:
            out["seed_aggregate"] = {k: seed_aggregate(v) for k, v in multi.items()}
            print(f"\n{'arm family':<22}{'seeds':>6}{'V mean':>10}"
                  f"{'between-seed se':>17}{'t_p':>10}{'sign concordant':>17}")
            for k, ag in out["seed_aggregate"].items():
                print(f"{k:<22}{ag['n_seeds']:>6}{ag['V_seed_mean']:>10.1f}"
                      f"{ag.get('V_between_seed_se', float('nan')):>17.1f}"
                      f"{ag.get('t_p', float('nan')):>10.2e}"
                      f"{str(ag['sign_concordant']):>17}")

    os.makedirs(os.path.join(ROOT, "runs"), exist_ok=True)
    opath = os.path.join(ROOT, "runs", f"{a.label}_rho{a.rho:g}.json")
    with open(opath, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n[stats] wrote {opath}")


if __name__ == "__main__":
    main()
