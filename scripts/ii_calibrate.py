"""scripts/ii_calibrate.py -- PHASE 2: does the wider action grid still learn?

Widening the grid to cover every regime SIGNAL-II contains reintroduces the problem that
registered decision R1 solved by narrowing it: a uniform initial policy orders up to the
grid midpoint and floods the chain. R1's fix was to narrow the grid, which works but
caps the regimes the study can reach -- and that cap cost SIGNAL-I a registered
hypothesis. The alternative is to decouple initialisation from grid width by biasing the
action head toward a sensible starting level, so the grid can be as wide as the
regimes require without the initial policy caring.

This script tests whether that works, on the reference cell only, before any campaign.

    python scripts/ii_calibrate.py --dry-run
    python scripts/ii_calibrate.py --workers 6
    python scripts/ii_calibrate.py --analyse

DECISION RULE, declared before any numbers are seen:

    Adopt the NARROWEST configuration that satisfies both:
      (a) mean final cost within 2% of configuration A (the SIGNAL-I control), and
      (b) grid ceiling >= the requirement from ii_fit_benchmarks.py.

    The 2% figure is a DECLARED CONVENTION, not a derived quantity. It is anchored to
    the magnitude Cachon and Fisher (2000) report for the value of shared information in
    their setting, on the reasoning that a change in the learner smaller than the effect
    the literature considers economically meaningful should not be treated as a change
    in the learner. Any threshold of this kind is a judgement; what makes it admissible
    is that it is fixed before the numbers exist, not that it is optimal.

    If no configuration satisfies (a), the grid is not the binding problem. Stop and
    diagnose rather than proceeding: a campaign built on a learner that got worse is
    worth less than no campaign.

Seeds 80-84 are a calibration space. They are never pooled with a confirmatory campaign
and no number from them is reported as a result.
"""
import argparse
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ii_local import run_jobs, verify                                  # noqa: E402

SEEDS = range(80, 85)
TOL = 0.02

# The four configurations. KEY NAMES ARE PLACEHOLDERS until ii_probe.py reports the real
# ones -- edit GRID_KEY / BINS_KEY / INIT_KEY below to whatever the probe found.
GRID_KEY = "s_grid_max"
BINS_KEY = "s_grid_bins"
INIT_KEY = "init_action_bias"

CONFIGS = {
    "A": {"label": "SIGNAL-I control, [0,100] x 41", "sets": ""},
    "B": {"label": "wide grid, uniform init", "sets": "{G}=200 {B}=81"},
    "C": {"label": "wide grid, biased init", "sets": "{G}=200 {B}=81 {I}=36"},
    "D": {"label": "wide grid, biased init, long warm-up",
          "sets": "{G}=200 {B}=81 {I}=36 warmup_episodes=3000"},
}
# init bias 36 units = mu(L+1) = 12 * 3, the stationary order-up-to level a chain needs
# just to cover its own pipeline. A policy that starts there is neither flooding nor
# starving on episode one.

ARMS = (("nocomm", "retailer_broadcast"), ("raw", "retailer_broadcast"))


def jobs_for(cfgs):
    out = []
    for c in cfgs:
        extra = CONFIGS[c]["sets"].format(G=GRID_KEY, B=BINS_KEY, I=INIT_KEY)
        for content, topo in ARMS:
            for s in SEEDS:
                tag = f"K{c}_{content}_s{s}"
                base = (f"tag={tag} content={content} seed={s} rho=0.9 beta=1.0 "
                        f"topology={topo} msg_scale=6.3 total_episodes=24000 "
                        f"demand_family=ar1")
                out.append({"tag": tag, "set": (base + " " + extra).strip()})
    return out


def analyse(cfgs):
    import numpy as np
    print("=" * 78)
    print("CALIBRATION RESULT -- reference cell only, seeds 80-84")
    print("=" * 78)
    res = {}
    for c in cfgs:
        for content, _ in ARMS:
            costs, evs, tops = [], [], []
            for s in SEEDS:
                gt = os.path.join(ROOT, "runs", f"K{c}_{content}_s{s}", "metrics_gate.csv")
                if not os.path.exists(gt):
                    continue
                rows = list(csv.DictReader(open(gt, encoding="utf-8")))
                fin = [r for r in rows if r.get("gate_cost") not in (None, "", "nan")]
                if not fin:
                    continue
                costs.append(float(fin[-1]["gate_cost"]))
                ev = [float(r["honest_ev"]) for r in rows
                      if r.get("honest_ev") not in (None, "", "nan")
                      and float(r.get("episode", 0)) >= 3000]
                if ev:
                    evs.append(min(ev))
            if costs:
                res[(c, content)] = {"cost": float(np.mean(costs)),
                                     "sd": float(np.std(costs, ddof=1)) if len(costs) > 1 else 0.0,
                                     "n": len(costs),
                                     "ev_min": min(evs) if evs else None}
    print(f"  {'cfg':4s} {'arm':8s} {'n':>2s} {'final cost':>12s} {'sd':>8s} "
          f"{'min honest EV':>14s}  {'vs A':>8s}")
    for c in cfgs:
        for content, _ in ARMS:
            r = res.get((c, content))
            if not r:
                print(f"  {c:4s} {content:8s}  -   (no data)")
                continue
            base = res.get(("A", content))
            rel = (r["cost"] / base["cost"] - 1) if base else float("nan")
            print(f"  {c:4s} {content:8s} {r['n']:2d} {r['cost']:12,.1f} {r['sd']:8,.1f} "
                  f"{(r['ev_min'] if r['ev_min'] is not None else float('nan')):14.3f}  "
                  f"{rel:+7.1%}")
    print()
    print("=" * 78)
    print("DECISION")
    print("=" * 78)
    sizing = os.path.join(ROOT, "docs", "II_SIZING.json")
    need = json.load(open(sizing, encoding="utf-8"))["ceiling"] if os.path.exists(sizing) else None
    if need:
        print(f"  ceiling required by the planned regimes: {need}")
    ok = []
    for c in cfgs:
        if c == "A":
            continue
        worst = None
        for content, _ in ARMS:
            r, base = res.get((c, content)), res.get(("A", content))
            if not r or not base:
                worst = None
                break
            rel = r["cost"] / base["cost"] - 1
            worst = rel if worst is None else max(worst, rel)
        if worst is None:
            print(f"  {c}: incomplete, cannot judge")
        elif worst <= TOL:
            print(f"  {c}: within {TOL:.0%} of the control (worst arm {worst:+.1%}) -- QUALIFIES")
            ok.append(c)
        else:
            print(f"  {c}: {worst:+.1%} worse than the control -- fails the {TOL:.0%} rule")
    print()
    if ok:
        print(f"  => adopt configuration {ok[0]} for SIGNAL-II "
              f"({CONFIGS[ok[0]]['label']})")
    else:
        print("  => NO configuration qualifies. The wider grid costs learning quality, so")
        print("     the grid is not the binding problem and widening it is not the fix.")
        print("     Do not proceed to the campaign; diagnose first. Likely candidates:")
        print("     entropy settings tuned for 41 bins spreading too thin over 81, or an")
        print("     initialisation that still floods. Report the numbers before deciding.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--configs", default="A,B,C,D")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    a = ap.parse_args()
    cfgs = [c.strip() for c in a.configs.split(",") if c.strip()]
    if a.analyse:
        analyse(cfgs)
        return
    jobs = jobs_for(cfgs)
    print(f"[calibrate] {len(cfgs)} configurations x {len(ARMS)} arms x {len(SEEDS)} "
          f"seeds = {len(jobs)} jobs")
    run_jobs(jobs, workers=a.workers, threads=a.threads, dry=a.dry_run)
    if not a.dry_run:
        verify([j["tag"] for j in jobs])
        print()
        print("now: python scripts/ii_calibrate.py --analyse")


if __name__ == "__main__":
    main()
