"""scripts/ii_pilot.py -- PHASE 3: feasibility and dispersion, not direction.

The pilot exists to answer three questions before twenty hours of pod time: does the
chosen configuration run clean, does the critic canary hold, and is the per-seed
DISPERSION of the timing contrast what the power calculation assumed.

It deliberately does not answer a fourth. The timing hypothesis was generated from
SIGNAL-I data -- the residual at rho=0 was observed, not predicted -- which makes it
exploratory. It becomes confirmatory only if it is registered before the data that tests
it exists. Looking at the sign of the lag effect here and then running the campaign would
mean testing a hypothesis on data that informed it, which is the exact failure SIGNAL-II
was designed to avoid.

So --analyse reports dispersion and withholds the effect. The withholding is enforced in
code rather than left to discipline, because discipline is what fails at 1am.

    python scripts/ii_pilot.py --dry-run
    python scripts/ii_pilot.py --workers 6
    python scripts/ii_pilot.py --analyse

Seeds 85-89 are a pilot space. They are never pooled with the confirmatory campaign.
"""
import argparse
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ii_local import run_jobs, verify                                  # noqa: E402

SEEDS = range(85, 90)
EXTRA = ""          # set to the winning configuration's --set string from ii_calibrate

# rho 0 is the regime that matters: there the conditional and unconditional benchmarks
# coincide, so the forecasting value of the signal is exactly zero and anything the
# channel carries must be demand leadtime in the sense of Hariharan and Zipkin (1995).
#
# The two lead times give the CROSSING test. A message broadcast on the period it is
# observed reaches the receiver tau = max(0, L - k) periods before the order carrying the
# same news, where k is the message lag. So at L = 2 a two-period lag leaves tau = 0 and
# the predicted value is exactly zero; at L = 4 the same lag still leaves tau = 2 and the
# predicted value is positive. The lag at which value vanishes should MOVE with the lead
# time, which is difficult to produce by any other mechanism.
CELLS = [(0.0, 2, "nocomm"), (0.0, 2, "raw"), (0.0, 2, "raw_lag2"),
         (0.0, 4, "nocomm"), (0.0, 4, "raw"), (0.0, 4, "raw_lag2")]
LEAD_KEY = "lead_time"      # placeholder: confirm the real key with ii_probe.py


def jobs():
    out = []
    for rho, L, content in CELLS:
        for s in SEEDS:
            tag = f"P_{content}_r{str(rho).replace('.', '')}_L{L}_s{s}"
            base = (f"tag={tag} content={content} seed={s} rho={rho:g} beta=1.0 "
                    f"topology=retailer_broadcast msg_scale=6.3 "
                    f"total_episodes=24000 demand_family=ar1 {LEAD_KEY}={L}")
            out.append({"tag": tag, "set": (base + " " + EXTRA).strip()})
    return out


def analyse():
    import numpy as np
    print("=" * 78)
    print("PILOT -- feasibility and dispersion")
    print("=" * 78)
    cost = {}
    for rho, L, content in CELLS:
        vals, evs = [], []
        for s in SEEDS:
            tag = f"P_{content}_r{str(rho).replace('.', '')}_L{L}_s{s}"
            gt = os.path.join(ROOT, "runs", tag, "metrics_gate.csv")
            if not os.path.exists(gt):
                continue
            rows = list(csv.DictReader(open(gt, encoding="utf-8")))
            fin = [r for r in rows if r.get("gate_cost") not in (None, "", "nan")]
            if fin:
                vals.append((s, float(fin[-1]["gate_cost"])))
            ev = [float(r["honest_ev"]) for r in rows
                  if r.get("honest_ev") not in (None, "", "nan")
                  and float(r.get("episode", 0)) >= 3000]
            if ev:
                evs.append(min(ev))
        cost[(rho, L, content)] = {"vals": dict(vals),
                                   "ev": min(evs) if evs else None}
    print(f"  {'rho':>5} {'L':>3} {'arm':10s} {'n':>2s} {'min honest EV':>14s}")
    for (rho, L, content), d in cost.items():
        ev = d["ev"]
        print(f"  {rho:>5g} {L:>3d} {content:10s} {len(d['vals']):2d} "
              f"{(ev if ev is not None else float('nan')):14.3f}")

    print()
    print("=" * 78)
    print("DISPERSION OF THE TIMING CONTRAST (sign withheld by design)")
    print("=" * 78)
    for rho, L in ((0.0, 2), (0.0, 4)):
        n_ = cost.get((rho, L, "nocomm"), {}).get("vals", {})
        r_ = cost.get((rho, L, "raw"), {}).get("vals", {})
        l_ = cost.get((rho, L, "raw_lag2"), {}).get("vals", {})
        common = sorted(set(n_) & set(r_) & set(l_))
        if len(common) < 3:
            print(f"  rho {rho:g}, L {L}: too few paired seeds ({len(common)})")
            continue
        v_raw = np.array([n_[s] - r_[s] for s in common])
        v_lag = np.array([n_[s] - l_[s] for s in common])
        d = v_raw - v_lag                      # the quantity the campaign will test
        sd = float(d.std(ddof=1))
        base = float(np.mean([n_[s] for s in common]))
        band = 0.02 * base
        need = 2
        from scipy import stats as st
        while need < 500 and st.t.ppf(0.975, need - 1) * sd / np.sqrt(need) > band:
            need += 1
        print(f"  rho {rho:g}, L {L}:  per-seed SD of the contrast = {sd:,.1f}")
        print(f"          no-sharing cost {base:,.0f}, so a 2% band is +/-{band:,.0f}")
        print(f"          n for the CI to fit inside the band: {need}")
        print(f"          (SIGNAL-II plans n = 15 -> "
              f"{'ADEQUATE' if need <= 15 else 'UNDERPOWERED, raise n or widen the band'})")
    print()
    print("  The magnitude and sign of the effect are deliberately not printed. This")
    print("  pilot sizes the confirmatory test; it does not preview its answer.")
    json.dump({"note": "pilot: dispersion only, seeds 85-89, not pooled"},
              open(os.path.join(ROOT, "docs", "II_PILOT.json"), "w", encoding="utf-8"),
              indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    a = ap.parse_args()
    if a.analyse:
        analyse()
        return
    j = jobs()
    print(f"[pilot] {len(CELLS)} cells x {len(SEEDS)} seeds = {len(j)} jobs")
    if not EXTRA:
        print("[pilot] NOTE: EXTRA is empty -- set it to the winning configuration's")
        print("        --set string from ii_calibrate before running for real.")
    run_jobs(j, workers=a.workers, threads=a.threads, dry=a.dry_run)
    if not a.dry_run:
        verify([x["tag"] for x in j])
        print()
        print("now: python scripts/ii_pilot.py --analyse")


if __name__ == "__main__":
    main()
