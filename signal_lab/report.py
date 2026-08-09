"""signal_lab/report.py -- verdicts and V, fail-closed by construction.

FAIL-CLOSED CONTRACT (the c7 lesson, kept for good): a run can only PASS by comparing
a FINITE eval mean against a FINITE in-project baseline. Anything else gets its own
non-PASS status -- NO-RUN (no run dir), NO-EVAL (no eval dump), NO-REF (baselines file
missing) -- one lookup shared by decision and display, no sentinels, and a non-zero
exit code unless every requested arm is PASS.

Binding bar: StaticBS (survival). CondBS is reported as the frontier. When a nocomm
run and comm runs share a seed base, the paired estimand V = C(nocomm) - C(comm) is
printed per arm with its per-episode pairing.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.environ.get("SIGNAL_REPORT_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))


EVAL_SEED_BASE = 10_000     # the canonical scoring space; must match evaluate.py


def _eval_costs(tag, rho, seed_base=EVAL_SEED_BASE):
    """Read ONLY the canonical eval dump.

    A previous version globbed seed*_rho<R>.json and merged every match. Because each
    dump keys episodes "0".."n-1", a diagnostic run at another seed base (e.g. the
    monitor space 60000) silently OVERWROTE the first episodes of the real eval and
    shifted the reported mean. Diagnostic dumps at other seed bases are ignored here;
    pass --seed-base to score one of them deliberately."""
    path = os.path.join(ROOT, "runs", tag, "eval", f"seed{seed_base}_rho{rho:g}.json")
    if not os.path.exists(path):
        return None
    vals = json.load(open(path))
    return np.array([float(v) for _, v in sorted(vals.items(), key=lambda kv: int(kv[0]))])


def verdict(tag, mean_cost, static_bar):
    if static_bar is None or not np.isfinite(static_bar):
        return "NO-REF", ("baselines file missing -- run signal_lab/baselines.py "
                          "first (fail-closed)")
    if mean_cost is None or not np.isfinite(mean_cost):
        return "NO-EVAL", "no deterministic eval dump -- run signal_lab/evaluate.py"
    if mean_cost < static_bar:
        return "PASS", f"beats StaticBS ({mean_cost:.1f} < {static_bar:.1f})"
    return "FAIL", f"does not beat StaticBS ({mean_cost:.1f} >= {static_bar:.1f})"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True, help="comma-separated run tags; the tag "
                    "containing 'nocomm' anchors the V column")
    ap.add_argument("--rho", type=float, default=0.9)
    ap.add_argument("--seed-base", type=int, default=EVAL_SEED_BASE,
                    dest="sb", help="scoring seed space (default: the canonical 10000)")
    a = ap.parse_args(argv)
    bpath = os.path.join(ROOT, "runs", f"baselines_rho{a.rho:g}.json")
    bars = json.load(open(bpath)) if os.path.exists(bpath) else None
    static_bar = float(bars["static_bs"]) if bars else None
    cond_bar = float(bars["cond_bs"]) if bars else None

    rows, statuses = [], []
    for tag in [t.strip() for t in a.arms.split(",")]:
        if not os.path.isdir(os.path.join(ROOT, "runs", tag)):
            rows.append((tag, None, "NO-RUN", "run dir missing"))
            statuses.append("NO-RUN")
            continue
        costs = _eval_costs(tag, a.rho, a.sb)
        mean = float(costs.mean()) if costs is not None else None
        st, why = verdict(tag, mean, static_bar)
        rows.append((tag, costs, st, why))
        statuses.append(st)

    print(f"== SIGNAL report  rho={a.rho:g}  "
          f"StaticBS={'%.1f' % static_bar if static_bar is not None else 'MISSING'}  "
          f"CondBS={'%.1f' % cond_bar if cond_bar is not None else 'MISSING'} ==")
    noc = next((c for t, c, _, _ in rows if "nocomm" in t and c is not None), None)
    for tag, costs, st, why in rows:
        mean = f"{float(costs.mean()):8.1f}" if costs is not None else "     ---"
        v = ""
        if noc is not None and costs is not None and "nocomm" not in tag:
            if len(costs) == len(noc):
                d = noc - costs
                v = f"   V={d.mean():+8.1f} (paired, n={len(d)}, se={d.std(ddof=1)/np.sqrt(len(d)):.1f})"
            else:
                v = "   V=UNPAIRED (episode counts differ -- fail-closed, no V)"
        print(f"   {tag:28s} {mean}   {st:8s} {why}{v}")
    if all(s == "PASS" for s in statuses):
        print(f"   all {len(statuses)} PASS.")
        return 0
    print("   not all PASS -- exit 1 (fail-closed).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
