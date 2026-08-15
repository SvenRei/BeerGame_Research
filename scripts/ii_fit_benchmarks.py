"""scripts/ii_fit_benchmarks.py -- PHASE 1: size the grid, and derive the preregistered
point prediction, both from analytic benchmarks alone.

Fitting benchmarks is coordinate descent over rollouts, not learning, so every regime
SIGNAL-II plans to contain can be priced in minutes without training anything. Two
things come out of it.

1. THE ACTION CEILING. The highest fitted order-up-to level across the planned regimes
   sets how wide the agent's action grid must be. SIGNAL-I did not perform this check;
   its grid stopped at 100 while two regimes required 102 and 106, which was found only
   after the campaign and cost a registered hypothesis.

2. THE POINT PREDICTION. Hariharan and Zipkin (1995, Management Science 41(10):1599-1607)
   show that advance demand information creates a "demand leadtime" that improves
   performance in precisely the same way a replenishment leadtime degrades it -- the two
   offset one for one. Milgrom and Roberts (1988) give the underlying substitution
   between information and inventory. That equivalence is a QUANTITATIVE prediction, not
   a direction: if sharing gives the receiver a demand leadtime of tau, its value should
   equal the cost saving from shortening the supply leadtime by tau, which the fitted
   benchmarks measure directly.

       predicted V(L, tau)  =  C*(L)  -  C*(L - tau)

   Computing that from benchmarks BEFORE the campaign turns SIGNAL-II from "we expect a
   positive residual" into a registered numerical prediction the data can miss.

    python scripts/ii_fit_benchmarks.py --dry-run
    python scripts/ii_fit_benchmarks.py
    python scripts/ii_fit_benchmarks.py --skip-fit      # re-read what is on disk

DECLARED DESIGN CHOICES (conventions, not derived quantities -- fixed here before any
number is seen, which is what makes them admissible):

  * ceiling margin 1.25 over the highest fitted level. A categorical policy with an
    entropy floor cannot concentrate its mass on the boundary bin, so a grid that merely
    REACHES the required level still cannot express it: SIGNAL-I's distributor needed
    102 against a ceiling of 100 and achieved a mean of 91. The margin is a convention
    chosen to leave room for that effect; it is NOT a theoretical quantity, and Phase 2
    verifies empirically that the policy does not saturate rather than trusting it.
  * resolution held at 2.5 units, matching SIGNAL-I, so bin width is not a confound
    between the two campaigns.
"""
import argparse
import glob
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MARGIN = 1.25          # declared convention, see module docstring
RESOLUTION = 2.5       # matches SIGNAL-I

# Regimes. The lead-time axis is the instrument for the demand-leadtime prediction: it
# varies delay per hop with the number of hops held constant.
#
# ECHELON COUNT IS DELIBERATELY NOT VARIED. Changing the number of stages changes the
# number of hops AND the cumulative delay to the factory at the same time, so an effect
# cannot be attributed to either. That is the same confound H-SOURCE already carries
# (relay differs from broadcast in hops and in delay), and adding a second confounded
# axis would not resolve the first. Lead time is the unconfounded instrument.
REGIMES = [
    {"name": "ar1_rho0.0_L2", "args": ["--rho", "0"]},
    {"name": "ar1_rho0.3_L2", "args": ["--rho", "0.3"]},
    {"name": "ar1_rho0.6_L2", "args": ["--rho", "0.6"]},
    {"name": "ar1_rho0.9_L2", "args": ["--rho", "0.9"]},
    {"name": "ar1_rho0.9_bh1", "args": ["--rho", "0.9", "--backorder-cost", "0.5"]},
    {"name": "ar1_rho0.9_bh4", "args": ["--rho", "0.9", "--backorder-cost", "2.0"]},
    {"name": "dr_poisson_L2", "args": ["--rho", "-1", "--demand-family", "dr_poisson"]},
    # the lead-time ladder. L = 0 is required as the tau = L endpoint of the prediction.
    {"name": "ar1_rho0.0_L0", "args": ["--rho", "0", "--lead-time", "0"], "new": True},
    {"name": "ar1_rho0.0_L1", "args": ["--rho", "0", "--lead-time", "1"], "new": True},
    {"name": "ar1_rho0.0_L3", "args": ["--rho", "0", "--lead-time", "3"], "new": True},
    {"name": "ar1_rho0.0_L4", "args": ["--rho", "0", "--lead-time", "4"], "new": True},
    {"name": "ar1_rho0.9_L0", "args": ["--rho", "0.9", "--lead-time", "0"], "new": True},
    {"name": "ar1_rho0.9_L1", "args": ["--rho", "0.9", "--lead-time", "1"], "new": True},
    {"name": "ar1_rho0.9_L3", "args": ["--rho", "0.9", "--lead-time", "3"], "new": True},
    {"name": "ar1_rho0.9_L4", "args": ["--rho", "0.9", "--lead-time", "4"], "new": True},
]


def fit(reg, dry):
    cmd = [sys.executable, "-m", "signal_lab.baselines"] + reg["args"] + \
          ["--fit-episodes", "12", "--eval-episodes", "30"]
    if dry:
        print("   ", " ".join(cmd))
        return
    print(f"[fit] {reg['name']}")
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        for t in (r.stderr or r.stdout).strip().splitlines()[-3:]:
            print("      ", t[:160])
        if reg.get("new"):
            print("       NOTE: uses --lead-time, an axis SIGNAL-I never varied. If the "
                  "flag is unrecognised, lead time is not settable and must be lifted "
                  "into the config before SIGNAL-II can test the demand-leadtime "
                  "prediction at all.")
        return
    for ln in r.stdout.splitlines():
        if "StaticBS" in ln:
            print("      ", ln.strip())


def collect():
    out = []
    for p in sorted(glob.glob(os.path.join(ROOT, "runs", "baselines_*.json"))):
        try:
            b = json.load(open(p, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        S = b.get("static_S") or []
        if not S:
            continue
        out.append({"file": os.path.basename(p), "S": [float(x) for x in S],
                    "max": max(float(x) for x in S), "rho": b.get("rho"),
                    "lead_time": b.get("lead_time"),
                    "static_bs": b.get("static_bs"), "cond_bs": b.get("cond_bs")})
    return out


def hz_prediction(rows):
    """The Hariharan-Zipkin (1995) equivalence, expressed in this study's quantities.

    A demand leadtime of tau offsets a replenishment leadtime of tau, so the value of a
    message arriving tau periods before the order carrying the same information is
    predicted to equal the cost difference between operating at leadtime L and at
    leadtime L - tau:

        V_predicted(L, tau) = C*(L) - C*(L - tau)

    Both terms are fitted benchmarks; nothing is estimated from a learner, so this is a
    prediction the campaign can fail. Because the message is broadcast in the period it
    is observed while the order takes L periods to convey the same news, tau = L for an
    undelayed message and tau = max(0, L - k) for one delayed by k. The prediction
    therefore has a CROSSING POINT that moves with L: value should vanish exactly where
    the message lag reaches the leadtime.
    """
    by = {}
    for r in rows:
        if r.get("lead_time") is None or r.get("rho") is None:
            continue
        by[(float(r["rho"]), int(r["lead_time"]))] = r
    if not by:
        return None
    out = []
    for (rho, L), r in sorted(by.items()):
        if r.get("static_bs") is None:
            continue
        for k in (0, 1, 2, 4):
            tau = max(0, L - k)
            ref = by.get((rho, L - tau))
            if ref is None or ref.get("static_bs") is None:
                continue
            out.append({"rho": rho, "L": L, "lag": k, "tau": tau,
                        "C_L": float(r["static_bs"]),
                        "C_Lminus": float(ref["static_bs"]),
                        "V_pred": float(r["static_bs"]) - float(ref["static_bs"])})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-fit", action="store_true")
    a = ap.parse_args()

    if not a.skip_fit:
        print("=" * 78)
        print("FITTING BENCHMARKS FOR EVERY PLANNED REGIME")
        print("=" * 78)
        for reg in REGIMES:
            fit(reg, a.dry_run)
        if a.dry_run:
            return

    rows = collect()
    if not rows:
        print("no baselines files found")
        return

    print()
    print("=" * 78)
    print("FITTED ORDER-UP-TO LEVELS")
    print("=" * 78)
    for r in rows:
        s = " / ".join(f"{x:.0f}" for x in r["S"])
        print(f"  {r['file']:38s} {s:26s} max {r['max']:6.0f}")

    hi = max(r["max"] for r in rows)
    ceiling = int(-(-(hi * MARGIN) // 50) * 50)
    bins = int(ceiling / RESOLUTION) + 1
    print()
    print("=" * 78)
    print("ACTION GRID (rule declared before the numbers were seen)")
    print("=" * 78)
    print(f"  highest fitted level : {hi:.0f}")
    print(f"  x {MARGIN} margin       : {hi * MARGIN:.0f}")
    print(f"  => grid [0, {ceiling}] with {bins} levels at {RESOLUTION} units")
    print(f"     (SIGNAL-I used [0, 100] with 41; "
          f"{'unchanged' if ceiling == 100 else f'{ceiling / 100:.1f}x wider'})")
    if ceiling > 100:
        print()
        print("  A wider grid means a uniform initial policy orders to the midpoint,")
        print(f"  {ceiling / 2:.0f} units -- the cold-start flood that registered decision R1")
        print("  narrowed the grid to prevent. Phase 2 tests whether biasing the action")
        print("  head's initial logits removes the problem without narrowing the grid.")

    pred = hz_prediction(rows)
    print()
    print("=" * 78)
    print("PREREGISTERED POINT PREDICTION -- Hariharan & Zipkin (1995)")
    print("=" * 78)
    if not pred:
        print("  Not computable: no benchmark carries a lead_time field. Either the")
        print("  lead-time regimes did not fit, or baselines.py does not record the lead")
        print("  time in its payload. Both must be fixed before SIGNAL-II can register a")
        print("  numerical prediction rather than a direction.")
    else:
        print("  A demand leadtime tau offsets a replenishment leadtime tau, so a message")
        print("  arriving tau periods ahead of the order carrying the same news is")
        print("  predicted to be worth C*(L) - C*(L-tau), both fitted analytically.")
        print()
        print(f"  {'rho':>5} {'L':>3} {'msg lag':>8} {'tau':>4} {'C*(L)':>10} "
              f"{'C*(L-tau)':>11} {'V predicted':>12}")
        for p_ in pred:
            print(f"  {p_['rho']:>5g} {p_['L']:>3d} {p_['lag']:>8d} {p_['tau']:>4d} "
                  f"{p_['C_L']:>10,.0f} {p_['C_Lminus']:>11,.0f} {p_['V_pred']:>12,.0f}")
        print()
        print("  The crossing point is the sharp test: where the message lag reaches the")
        print("  leadtime, tau = 0 and the predicted value is exactly zero. That point")
        print("  MOVES with L, which is hard to reproduce by any mechanism other than the")
        print("  one being claimed.")

    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    json.dump({"rows": rows, "highest": hi, "margin": MARGIN, "ceiling": ceiling,
               "bins": bins, "hz_prediction": pred},
              open(os.path.join(ROOT, "docs", "II_SIZING.json"), "w",
                   encoding="utf-8"), indent=1)
    print()
    print("[sizing] wrote docs/II_SIZING.json")


if __name__ == "__main__":
    main()
