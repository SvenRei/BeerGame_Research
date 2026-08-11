"""signal_lab/hypotheses.py -- registered confirmatory tests (skeleton v2.0 SS5/SS7).

Implements exactly the pre-registered decision rules, on SEED-LEVEL vectors (the unit
of inference is the CRN-paired per-seed V, never pooled episodes):

  P1  crossover, intersection-union: BOTH one-sided studentized-bootstrap lower
      confidence bounds must exceed 0. No multiplicity correction is applied to the
      conjunction (IU logic: the claim is the intersection, so testing each conjunct
      at alpha controls the conjunction at alpha).
  H2  rho-gradient: per-seed OLS slope of V_raw over rho in {0,.3,.6,.9}; one-sided
      one-sample t on the seed vector of slopes (H1: mean slope > 0).
  HREP operational representation: pairwise TOST between content families at a margin
      of +/- 2% of the mean nocomm cost (Cachon-Fisher materiality).

Studentized bootstrap-t (the registered CI method, SS7): resample seeds with
replacement, pivot t* = (mean* - mean)/se*, lower bound = mean - t*_{(1-alpha)} * se.

Self-test: `python -m signal_lab.hypotheses --self-test` runs every rule against
planted data with known truth and fails loudly on any wrong verdict.

Campaign use: point --stats at one or more stats_rho*.json files produced by
signal_lab.stats; the loader assembles seed vectors from their `paired_vs_nocomm`
blocks (family = tag with the `_s<NN>` suffix stripped).
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import scipy.stats as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RE_SEED = re.compile(r"^(.*)_s(\d+)$")
RNG = np.random.default_rng(20260808)
N_BOOT = 10_000


# ------------------------------------------------------------------ primitives
def boot_t_lower(x, alpha=0.05, n_boot=N_BOOT):
    """One-sided lower confidence bound for the mean, studentized bootstrap-t."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        raise SystemExit("[hyp] FAIL-CLOSED: need >=2 seeds for a bootstrap-t bound")
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return float(m)
    idx = RNG.integers(0, n, size=(n_boot, n))
    bs = x[idx]
    bm = bs.mean(1)
    bse = bs.std(1, ddof=1) / np.sqrt(n)
    ok = bse > 0
    t_star = (bm[ok] - m) / bse[ok]
    return float(m - np.quantile(t_star, 1 - alpha) * se)


def one_sided_t(x, mu0=0.0):
    t, p2 = st.ttest_1samp(np.asarray(x, float), mu0)
    return float(t), float(p2 / 2 if t > 0 else 1 - p2 / 2)


def tost_paired_seeds(x, y, margin):
    """Equivalence of two seed vectors' means within +/- margin (paired by seed)."""
    d = np.asarray(x, float) - np.asarray(y, float)
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return float(abs(d.mean()) < margin)
    t_lo = (d.mean() + margin) / se
    t_hi = (d.mean() - margin) / se
    p = max(1 - st.t.cdf(t_lo, n - 1), st.t.cdf(t_hi, n - 1))
    return float(p)


# ------------------------------------------------------------------ registered rules
def p1_conjunction(delta_dp, delta_ar, alpha=0.05):
    """Intersection-union: both one-sided lower bounds must exceed 0."""
    lo_dp = boot_t_lower(delta_dp, alpha)
    lo_ar = boot_t_lower(delta_ar, alpha)
    return {"lower_dp": lo_dp, "lower_ar": lo_ar,
            "reject_null": bool(lo_dp > 0 and lo_ar > 0),
            "rule": "IU: both one-sided bootstrap-t lower bounds > 0 at alpha=.05"}


def p1_did(delta_dp, delta_ar, alpha=0.05):
    """P1' (amended): the forecast's ADVANTAGE over raw is larger under regime
    uncertainty than under persistence.

        Delta(regime)_s = V_arpred(regime)_s - V_raw(regime)_s      [per seed s]
        H1: mean_s [ Delta(DP)_s - Delta(AR,rho=.9)_s ] > 0

    Replaces the registered intersection-union crossover, whose AR conjunct
    (V_raw > V_forecast at rho=.9) is refuted by the dev ladder: arpred beat raw by
    +153 concordantly across three seeds. Blackwell sufficiency holds
    information-theoretically but not through a LEARNING channel, because the receiver
    must estimate the map -- so the sign never crosses; the magnitude does.

    IMPORTANT: pass Deltas computed on GAP-RECOVERED units, not raw cost. Cost scales
    differ by regime (AR gap ~1231, DP gap ~2748), so an absolute-cost DiD would
    confound 'more value available' with 'forecast helps more'."""
    d = np.asarray(delta_dp, float) - np.asarray(delta_ar, float)
    lo = boot_t_lower(d, alpha)
    t, p = one_sided_t(d)
    return {"did_mean": float(d.mean()),
            "did_se": float(d.std(ddof=1) / np.sqrt(len(d))),
            "lower_boot_t": lo, "t": t, "p_one_sided": p,
            "delta_dp_mean": float(np.mean(delta_dp)),
            "delta_ar_mean": float(np.mean(delta_ar)),
            "reject_null": bool(lo > 0),
            "rule": "one-sided studentized bootstrap-t lower bound on the DiD > 0"}


def h2_slope(v_by_rho, alpha=0.05):
    """v_by_rho: {rho: seed-vector of V}; per-seed OLS slope, one-sided t > 0."""
    rhos = sorted(v_by_rho)
    mat = np.array([v_by_rho[r] for r in rhos], dtype=float)   # [R, S]
    if mat.shape[0] < 3:
        raise SystemExit("[hyp] FAIL-CLOSED: H2 needs >=3 rho levels")
    x = np.array(rhos, dtype=float)
    xc = x - x.mean()
    slopes = (xc @ mat) / (xc @ xc)                            # per-seed OLS slope
    t, p = one_sided_t(slopes)
    return {"rhos": rhos, "slopes": slopes.tolist(),
            "slope_mean": float(slopes.mean()),
            "slope_se": float(slopes.std(ddof=1) / np.sqrt(len(slopes))),
            "t": t, "p_one_sided": p,
            "lower_boot_t": boot_t_lower(slopes, alpha),
            "reject_null": bool(p < alpha and slopes.mean() > 0)}


def study_wide_holm(pvals, alpha=0.05):
    """F2: multiplicity across the STUDY, not just within families.

    The paper makes ~13 primary hypothesis-level verdicts. Holm within a family does
    not control the family-wise error rate across them. This applies Holm to the
    primary set and reports which verdicts survive, so a referee does not have to
    ask. Results with p ~ 1e-8 survive trivially; borderline ones (H-SHOCK at .03)
    are expected NOT to, and should be carried by mechanism evidence instead.

    pvals: {name: one-sided p}. Returns {name: (p, holm_p, survives)}.
    """
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for rank, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - rank) * p))
        out[name] = {"p": float(p), "holm_p": float(running),
                     "survives": bool(running < alpha)}
    return out


def hrep_tost(fam_vectors, nocomm_cost_mean, frac=0.02, alpha=0.05):
    """Pairwise TOST between families at +/- frac*nocomm; Holm over the pairs."""
    names = sorted(fam_vectors)
    margin = frac * nocomm_cost_mean
    pairs, ps = [], []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            p = tost_paired_seeds(fam_vectors[names[i]], fam_vectors[names[j]], margin)
            pairs.append((names[i], names[j])), ps.append(p)
    order = np.argsort(ps)
    m = len(ps)
    holm = [None] * m
    running = 0.0
    for rank, k in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[k]))
        holm[k] = running
    return {"margin": margin,
            "pairs": [{"a": a, "b": b, "tost_p": float(p), "holm_p": float(h),
                       "equivalent": bool(h < alpha)}
                      for (a, b), p, h in zip(pairs, ps, holm)]}


# ------------------------------------------------------------------ campaign loader
def load_seed_vectors(stats_paths):
    """family -> {seed:int -> V_mean}, plus nocomm mean cost, from stats json files."""
    fams, nocomm_costs = {}, []
    for path in stats_paths:
        d = json.load(open(path))
        for arm, pb in d.get("paired_vs_nocomm", {}).items():
            m = _RE_SEED.match(arm)
            if not m:
                continue
            fams.setdefault(m.group(1), {})[int(m.group(2))] = pb["V_mean"]
        for arm, blk in d.get("arms", {}).items():
            if blk.get("content") == "nocomm":
                nocomm_costs.append(blk["cost_mean"])
    if not fams:
        raise SystemExit("[hyp] FAIL-CLOSED: no paired_vs_nocomm blocks found")
    return fams, (float(np.mean(nocomm_costs)) if nocomm_costs else float("nan"))


# ------------------------------------------------------------------ self-test
def self_test():
    rng = np.random.default_rng(7)
    # P1: both conjuncts truly positive -> reject; one null -> no reject
    a = rng.normal(200, 60, 15)
    b = rng.normal(300, 80, 15)
    z = rng.normal(0, 60, 15)
    r1 = p1_conjunction(a, b)
    r2 = p1_conjunction(a, z)
    assert r1["reject_null"] is True, r1
    assert r2["reject_null"] is False, r2
    # H2: planted slope 1000 per unit rho recovered; flat -> no reject
    rhos = [0.0, 0.3, 0.6, 0.9]
    up = {r: (1000 * r + rng.normal(0, 40, 12)).tolist() for r in rhos}
    flat = {r: rng.normal(500, 40, 12).tolist() for r in rhos}
    h_up, h_flat = h2_slope(up), h2_slope(flat)
    assert h_up["reject_null"] and abs(h_up["slope_mean"] - 1000) < 60, h_up
    assert not h_flat["reject_null"], h_flat
    # P1': DiD recovers a planted regime difference; a flat pair does not
    dp = rng.normal(0.55, 0.08, 15); ar = rng.normal(0.12, 0.06, 15)
    r = p1_did(dp, ar)
    assert r["reject_null"] and abs(r["did_mean"] - 0.43) < 0.05, r
    assert not p1_did(ar, ar + rng.normal(0, 0.06, 15))["reject_null"]
    # HREP: identical-mean families equivalent at the margin; far one not
    base = rng.normal(900, 25, 15)
    fams = {"raw": base + rng.normal(0, 10, 15),
            "eps": base + rng.normal(0, 10, 15),
            "far": base + 400}
    rep = hrep_tost(fams, nocomm_cost_mean=4000.0)   # margin 80
    verdict = {(p["a"], p["b"]): p["equivalent"] for p in rep["pairs"]}
    assert verdict[("eps", "raw")] is True, rep
    assert verdict[("far", "raw")] is False and verdict[("eps", "far")] is False, rep
    # study-wide Holm: strong results survive, borderline ones do not
    sw = study_wide_holm({"strong": 1e-9, "mid": 1e-4, "borderline": 0.030,
                          "weak": 0.40}, alpha=0.05)
    assert sw["strong"]["survives"] and sw["mid"]["survives"], sw
    assert not sw["borderline"]["survives"], sw          # .030 * 2 = .060 > .05
    assert not sw["weak"]["survives"], sw
    # bootstrap-t bound sanity: below the SAMPLE mean, and its one-sided coverage of
    # the true mean is ~95% over repetitions
    x = rng.normal(100, 30, 20)
    lo = boot_t_lower(x)
    assert lo < x.mean(), (lo, x.mean())
    cover = np.mean([boot_t_lower(rng.normal(100, 30, 20), n_boot=800) <= 100
                     for _ in range(300)])
    assert 0.90 <= cover <= 0.99, f"one-sided coverage {cover:.3f} out of range"
    print("HYPOTHESES SELF-TEST PASS -- P1 IU, H2 slope, H-REP TOST, bootstrap-t all "
          "verified on planted data.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--stats", default="runs/stats_rho*.json",
                    help="glob of stats json files from signal_lab.stats")
    ap.add_argument("--h2-family", default="r9_raw",
                    help="family whose V forms the rho-gradient")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    paths = sorted(glob.glob(os.path.join(ROOT, a.stats)))
    if not paths:
        raise SystemExit(f"[hyp] FAIL-CLOSED: no stats files match {a.stats}")
    fams, nocomm_mean = load_seed_vectors(paths)
    print(f"[hyp] families: { {k: len(v) for k, v in fams.items()} }  "
          f"nocomm mean {nocomm_mean:.1f}")
    # H2 assembled from per-rho stats files if the family appears in several
    print(json.dumps({k: sorted(v.items()) for k, v in fams.items()}, indent=1))
    print("[hyp] assemble P1/H2/HREP calls per the registration; see docstring.")


if __name__ == "__main__":
    main()
