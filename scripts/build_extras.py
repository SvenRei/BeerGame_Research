"""scripts/build_extras.py -- the trajectory-derived half of the report.

runs/RESULTS.csv gives one row per arm. Three results need per-STEP or per-CHECKPOINT
detail that no summary row can carry:

    H-ECHELON   cost decomposed by stage          <- eval/*_traj.json  (cost_agent)
    H-BUDGET    value at each training milestone  <- eval/*__ckpt_budget*.json
    latency     weeks until each stage responds   <- eval/*_rho-3_traj.json (orders)

plus the shock trace, the backlog panels, the cost split and the message-response
curves. This script extracts them once into docs/extras.json so build_report.py never
has to open a 1 GB trajectory tree.

    python scripts/build_extras.py                 # after a campaign, from runs/
    python scripts/build_extras.py --runs /path/to/runs

Every block stores the PER-SEED vector alongside the mean, so the report can show a
full inference table (SE, CI, d_z, t, p, Wilcoxon, concordance) for these results
exactly as it does for the ones computed from the CSV. Blocks whose inputs are missing
are skipped with a note rather than failing: a partial extras file still builds a
report, it just omits those figures.
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import stats as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = range(30, 45)
G = "C_ar1_r09_{c}_reta_b10_s{s}"


def _load(runs, tag, label, suffix=""):
    p = os.path.join(runs, tag, "eval", f"seed10000_rho{label}{suffix}_traj.json")
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p, encoding="utf-8"))
    except (ValueError, OSError):
        return None


def echelon(runs, seeds):
    """Cost reduction per stage, per seed, for each content."""
    out, per = {}, {}
    for c in ("raw", "dhatc", "arpred", "learned"):
        rows = []
        for s in seeds:
            a = _load(runs, G.format(c="nocomm", s=s), "0.9")
            b = _load(runs, G.format(c=c, s=s), "0.9")
            if not a or not b:
                continue
            na = np.array([r["cost_agent"] for r in a["per_episode"]]).mean(0)
            nb = np.array([r["cost_agent"] for r in b["per_episode"]]).mean(0)
            rows.append(na - nb)
        if len(rows) < 3:
            continue
        A = np.array(rows)
        out[c] = {"mean": A.mean(0).round(1).tolist(),
                  "se": (A.std(0, ddof=1) / np.sqrt(len(A))).round(1).tolist()}
        per[c] = A.T.round(1).tolist()          # [stage][seed]
    if "raw" not in out:
        return None
    A = np.array(per["raw"]).T
    d = A[:, 1:].sum(1) - A[:, 0]
    return {"by_content": out, "per_seed": per,
            "upstream_vs_retailer": {
                "diff": round(float(d.mean()), 1),
                "se": round(float(d.std(ddof=1) / np.sqrt(len(d))), 1),
                "p": float(st.ttest_1samp(d, 0).pvalue)},
            "share": (A.mean(0) / A.mean(0).sum()).round(3).tolist()}


def budget(runs, seeds):
    """V at each training milestone, no-sharing arm taken at the SAME milestone so
    planning capability is held equal on both sides of the comparison."""
    mile = ((2000, "__ckpt_budget2000"), (6000, "__ckpt_budget6000"),
            (12000, "__ckpt_budget12000"), (24000, ""))
    lv, ok = {}, []
    for m, suf in mile:
        vals = []
        for s in seeds:
            a = _load(runs, G.format(c="nocomm", s=s), "0.9", suf)
            b = _load(runs, G.format(c="raw", s=s), "0.9", suf)
            if not a or not b:
                continue
            ca = np.mean([r["team_cost"] for r in a["per_episode"]])
            cb = np.mean([r["team_cost"] for r in b["per_episode"]])
            vals.append(ca - cb)
        if len(vals) < 3:
            continue
        v = np.array(vals)
        lv[str(m)] = {"V": round(float(v.mean()), 1),
                      "se": round(float(v.std(ddof=1) / np.sqrt(len(v))), 1),
                      "n": len(v), "concordant": bool(all(v > 0)),
                      "per_seed": v.round(1).tolist()}
        ok.append((m, v))
    if len(ok) < 3:
        return None
    n = min(len(v) for _, v in ok)
    x = np.log2(np.array([m for m, _ in ok], dtype=float))
    xc = x - x.mean()
    sl = (xc @ np.array([v[:n] for _, v in ok])) / (xc @ xc)
    return {"levels": lv, "slope": round(float(sl.mean()), 1),
            "slope_se": round(float(sl.std(ddof=1) / np.sqrt(len(sl))), 1),
            "slope_per_seed": sl.round(1).tolist(),
            "p": float(st.ttest_1samp(sl, 0).pvalue)}


def latency(runs, seeds, shock=24):
    """Weeks until each stage raises its orders after the shock, at three thresholds."""
    def one(tag, thr):
        d = _load(runs, tag, "-3")
        if not d:
            return None
        O = np.array([r["orders"] for r in d["per_episode"]]).mean(0)
        return [next((w - shock for w in range(shock, len(O)) if O[w, i] > thr), 99)
                for i in range(4)]
    out = {}
    for thr in (12, 14, 16):
        N = [one(G.format(c="nocomm", s=s), thr) for s in seeds]
        R = [one(G.format(c="raw", s=s), thr) for s in seeds]
        N = [x for x in N if x]
        R = [x for x in R if x]
        if not N or not R:
            return None
        out[str(thr)] = {"nocomm": np.median(N, 0).astype(int).tolist(),
                         "raw": np.median(R, 0).astype(int).tolist()}
    return out


def trace(runs, seeds, w0=18, shock=24, span=26):
    """Factory orders and realised demand through the shock, averaged over seeds."""
    out = {"w0": w0, "shock": shock}
    for arm, c in (("nocomm", "nocomm"), ("raw", "raw")):
        rows = [_load(runs, G.format(c=c, s=s), "-3") for s in seeds]
        rows = [r for r in rows if r]
        if not rows:
            return None
        O = np.mean([np.array([x["orders"] for x in r["per_episode"]]).mean(0)
                     for r in rows], 0)
        out[arm] = O[w0:w0 + span, 3].round(1).tolist()
    rows = [_load(runs, G.format(c="nocomm", s=s), "-3") for s in seeds]
    rows = [r for r in rows if r]
    D = np.mean([np.array([x["demand"] for x in r["per_episode"]]).mean(0)
                 for r in rows], 0)
    out["demand"] = D[w0:w0 + span].round(1).tolist()
    out["n"] = len(rows)
    return out


def shock_state(runs, seeds, w0=18, shock=24, span=26):
    """Inventory and backlog at every stage through the shock."""
    out = {"w0": w0, "shock": shock}
    for arm, c in (("nocomm", "nocomm"), ("raw", "raw")):
        rows = [_load(runs, G.format(c=c, s=s), "-3") for s in seeds]
        rows = [r for r in rows if r]
        if not rows:
            return None
        inv = np.mean([np.array([x["inv"] for x in r["per_episode"]]).mean(0)
                       for r in rows], 0)
        bak = np.mean([np.array([x["back"] for x in r["per_episode"]]).mean(0)
                       for r in rows], 0)
        out[arm] = {"inv": inv[w0:w0 + span].round(1).tolist(),
                    "back": bak[w0:w0 + span].round(1).tolist()}
    return out


def cost_split(runs, seeds):
    """Holding versus backorder cost at every stage."""
    out = {}
    for arm, c in (("nocomm", "nocomm"), ("raw", "raw")):
        rows = [_load(runs, G.format(c=c, s=s), "0.9") for s in seeds]
        rows = [r for r in rows if r]
        if not rows:
            return None
        h = np.mean([np.array([x["cost_hold"] for x in r["per_episode"]]).mean(0)
                     for r in rows], 0)
        b = np.mean([np.array([x["cost_back"] for x in r["per_episode"]]).mean(0)
                     for r in rows], 0)
        out[arm] = {"hold": h.round(1).tolist(), "back": b.round(1).tolist()}
    return out


def response(runs, seeds):
    """How far a receiver moves its order-up-to level per unit of message received."""
    tags = {"retailer_broadcast": "C_ar1_r09_raw_reta_b10_s{s}",
            "upstream_only": "C_ar1_r09_raw_upst_b10_s{s}",
            "downstream_only": "C_ar1_r09_raw_down_b10_s{s}"}
    out = {}
    for lab, pat in tags.items():
        sl, M, S = [], [], []
        for s in seeds:
            d = _load(runs, pat.format(s=s), "0.9")
            if not d:
                continue
            m = np.array([r["msg0"] for r in d["per_episode"]])[:, 1:, 1].ravel()
            a = (np.array([r["actions"] for r in d["per_episode"]])[:, 1:, 1] * 2.5).ravel()
            ok = m > 0
            if ok.sum() > 50:
                sl.append(np.polyfit(m[ok], a[ok], 1)[0])
                M.append(m[ok])
                S.append(a[ok])
        if not sl:
            out[lab] = None
            continue
        m, a = np.concatenate(M), np.concatenate(S)
        bins = np.linspace(np.percentile(m, 2), np.percentile(m, 98), 9)
        pts = [[round(float(m[(m >= bins[i]) & (m < bins[i + 1])].mean()), 2),
                round(float(a[(m >= bins[i]) & (m < bins[i + 1])].mean()), 1)]
               for i in range(len(bins) - 1)
               if ((m >= bins[i]) & (m < bins[i + 1])).sum() > 50]
        out[lab] = {"pts": pts, "slope": round(float(np.mean(sl)), 2),
                    "se": round(float(np.std(sl, ddof=1) / np.sqrt(len(sl))), 2),
                    "n": len(sl)}
    return out


def _rho_tag(rho):
    """Campaign tag for the nocomm arm of an AR(1) rho group. The tag strips the dot
    (r09) while the eval dump keeps it (rho0.9) -- the two conventions are not the same
    string and mixing them is why a block silently finds nothing."""
    return "C_ar1_r{r}_nocomm_reta_b10_s{{s}}".format(r=str(rho).replace(".", ""))


RHOS = (0, 0.3, 0.6, 0.9)


def demand_shapes(runs, seeds, weeks=150):
    """One demand realisation per persistence level, ~150 weeks, each from its OWN seed:
    sharing a seed across panels makes the four look like copies and hides the structure
    the figure exists to show. Episodes are concatenated to reach the window."""
    out = {}
    for j, rho in enumerate(RHOS):
        s = list(seeds)[j % len(list(seeds))]
        d = _load(runs, _rho_tag(rho).format(s=s), f"{rho:g}")
        if not d:
            continue
        seq = []
        for ep in d["per_episode"]:
            seq.extend(float(x) for x in ep["demand"])
            if len(seq) >= weeks:
                break
        if len(seq) < 20:
            continue
        out[f"{float(rho):.1f}"] = [round(x, 1) for x in seq[:weeks]]
    return out or None


def demand_stats(runs, seeds, max_ep=200):
    """Dimensions of the demand process measured over many episodes rather than read off
    the single paths in the figure above: a 150-week path can show range and persistence
    out of order by luck, and the table is what the reader should trust."""
    out, sd0 = {}, None
    for rho in RHOS:
        eps = []
        for s in seeds:
            d = _load(runs, _rho_tag(rho).format(s=s), f"{rho:g}")
            if not d:
                continue
            for ep in d["per_episode"]:
                eps.append(np.asarray(ep["demand"], float))
                if len(eps) >= max_ep:
                    break
            if len(eps) >= max_ep:
                break
        if len(eps) < 10:
            continue
        flat = np.concatenate(eps)
        lags = [np.corrcoef(e[:-1], e[1:])[0, 1] for e in eps
                if e.std() > 0 and len(e) > 2]
        ptp = float(np.mean([e.max() - e.min() for e in eps]))

        def longest_run(e):
            """Longest stretch on one side of the episode's own mean -- the plainest
            measure of 'demand drifts in swings' that does not assume a model."""
            side, best, cur = e > e.mean(), 0, 0
            for k in range(len(side)):
                cur = cur + 1 if k and side[k] == side[k - 1] else 1
                best = max(best, cur)
            return best
        run = float(np.mean([longest_run(e) for e in eps]))
        sd = float(flat.std(ddof=1))
        if rho == 0:
            sd0 = sd
        out[f"{float(rho):.1f}"] = {
            "mean": round(float(flat.mean()), 1), "sd": round(sd, 2),
            "lag1": round(float(np.mean(lags)), 2) if lags else 0.0,
            "ptp": round(ptp, 1), "run": round(run, 1), "n_ep": len(eps),
            "sd_mult": round(sd / sd0, 2) if sd0 else 1.0}
    return out or None


def shock_orders(runs, seeds, w0=18, shock=24, span=26):
    """What each stage orders through the unannounced doubling, as a median with a
    across-seed band, plus each seed's pre-shock level and peak. The band is the point:
    a median alone cannot show whether sharing narrows the disagreement between runs."""
    out = {"w0": w0, "shock": shock}
    for arm, c in (("nocomm", "nocomm"), ("raw", "raw")):
        rows = [_load(runs, G.format(c=c, s=s), "-3") for s in seeds]
        rows = [r for r in rows if r]
        if not rows:
            return None
        per = np.array([np.array([x["orders"] for x in r["per_episode"]]).mean(0)
                        for r in rows])                      # [seed][week][stage]
        win = per[:, w0:w0 + span, :]
        out[arm] = {
            "med": np.median(win, 0).round(1).tolist(),
            "lo": np.percentile(win, 25, axis=0).round(1).tolist(),
            "hi": np.percentile(win, 75, axis=0).round(1).tolist(),
            "peak": per[:, shock:, :].max(1).round(1).tolist(),
            "base": per[:, w0:shock, :].mean(1).round(1).tolist(),
            "n": len(rows)}
    rows = [_load(runs, G.format(c="nocomm", s=s), "-3") for s in seeds]
    rows = [r for r in rows if r]
    D = np.mean([np.array([x["demand"] for x in r["per_episode"]]).mean(0)
                 for r in rows], 0)
    out["demand"] = D[w0:w0 + span].round(1).tolist()
    return out


def baseline_traj(runs, seeds):
    """The fitted base-stock rule and the trained no-sharing agents on the SAME demand,
    so the bullwhip comparison is like-for-like. The static half is re-rolled here
    rather than stored, because a stored trajectory can drift from the fitted levels
    that runs/baselines_rho0.9.json actually records."""
    lp = os.path.join(ROOT, "runs", "baselines_rho0.9.json")
    d = _load(runs, G.format(c="nocomm", s=list(seeds)[0]), "0.9")
    if not d or not os.path.exists(lp):
        return None
    learned = np.array(d["per_episode"][0]["orders"], float)
    try:
        sys.path.insert(0, ROOT)
        from env.beer_game import AGENTS, BeerGame, N_AGENTS   # noqa: F401
        bl = json.load(open(lp, encoding="utf-8"))
        S = np.array(bl["static_S"], float)
        cfg = {"holding_cost": 0.5, "backorder_cost": 1.0, "demand_family": "ar1",
               "ar1_rho": 0.9, "ar1_mu": 12.0, "ar1_sigma": 3.0}
        env = BeerGame(cfg)
        obs = env.reset(seed=10_000)
        orders, inv, done = [], [], False
        while not done:
            ip = np.array([BeerGame.inventory_position(obs[i])
                           for i in range(N_AGENTS)])
            o = np.clip(np.round(S - ip), 0, env.max_order)
            orders.append(o.tolist())
            inv.append([float(BeerGame.inventory_position(obs[i]))
                        for i in range(N_AGENTS)])
            obs, _c, done, _i = env.step(o)
    except Exception as e:
        print(f"[extras] baseline_traj: static rollout unavailable ({type(e).__name__}: "
              f"{e}); the panel needs the env importable from ROOT")
        return None
    O = np.array(orders, float)
    n = min(len(O), len(learned))
    # The caption calls this an order VARIANCE ratio, so it must be variance, not the
    # standard deviation. Normalised to the retailer so the displayed number reads
    # directly as factory-to-retailer amplification, matching the bw_* columns in
    # RESULTS.csv. NOTE this panel is ONE episode from ONE seed -- it is an
    # illustration of the mechanism, not the inferential bullwhip result, which is the
    # 15-seed bw_retailer..bw_manufacturer columns.
    def ratio(M):
        base = max(1e-9, float(M[:, 0].var()))
        return (M.var(0) / base).round(2).tolist()
    return {"orders": O[:n].round(1).tolist(),
            "inv": np.array(inv, float)[:n].round(1).tolist(),
            "learned": learned[:n].round(1).tolist(),
            "var_static": ratio(O[:n]),
            "var_learned": ratio(learned[:n]),
            "single_episode": True, "seed": int(list(seeds)[0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "extras.json"))
    ap.add_argument("--seed-start", type=int, default=30)
    ap.add_argument("--n-seeds", type=int, default=15)
    a = ap.parse_args()
    seeds = range(a.seed_start, a.seed_start + a.n_seeds)
    blocks = {"echelon": echelon, "budget": budget, "latency": latency,
              "trace": trace, "shock_state": shock_state,
              "cost_split": cost_split, "response": response,
              "demand_shapes": demand_shapes, "demand_stats": demand_stats,
              "shock_orders": shock_orders, "baseline_traj": baseline_traj}
    out = {}
    for name, fn in blocks.items():
        try:
            r = fn(a.runs, seeds)
        except Exception as e:                       # never lose the whole file
            print(f"[extras] {name}: FAILED ({type(e).__name__}: {e})")
            continue
        if r is None:
            print(f"[extras] {name}: skipped (inputs missing)")
        else:
            out[name] = r
            print(f"[extras] {name}: ok")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print(f"[extras] wrote {a.out} ({len(out)}/{len(blocks)} blocks)")
    # A count of what THIS script knows how to build is not a count of what the report
    # needs. The previous version produced 7 of 7 and the report still lost four
    # figures, because four of its consumers were never generated at all.
    missing = [b for b in blocks if b not in out]
    if missing:
        print(f"[extras] MISSING -- these figures will be omitted from the report: "
              f"{', '.join(missing)}")


if __name__ == "__main__":
    main()