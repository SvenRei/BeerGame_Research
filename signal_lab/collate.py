"""signal_lab/collate.py -- ONE results sheet from the whole campaign.

The campaign writes one stats_*.json per analysis group (family x rho x beta x clip,
plus the OOD labels). That is correct for fail-closed analysis but useless for reading.
This walks every stats file and emits:

    runs/RESULTS.csv   one row per (group, arm)  -- everything, machine-readable
    runs/RESULTS.md    grouped tables + the seed-level aggregates -- human-readable

Columns: group, regime, rho label, beta, clip, arm family, seed, cost, se, CVaR,
bullwhip per echelon, ready rate, V vs matched nocomm (+ se, dz, CI, Holm p),
V vs StaticBS, gap recovered, listening deltas (zeroed / shuffled), signaling r.

Never invents a number: a missing block becomes an empty cell, and the group label is
carried through verbatim so a row can always be traced back to its stats file.

    python -m signal_lab.collate                 # all runs/stats_*.json
    python -m signal_lab.collate --out mysheet   # custom stem
"""
import argparse
import csv
import glob
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RE_SEED = re.compile(r"^(.*)_s(\d+)$")
# analysis labels that are NOT an AR(1) rho
_LABELS = {-1.0: "dr_poisson (regime uncertainty)", -3.0: "black_swan (OOD)",
           -4.0: "extreme_chaos (OOD)", -5.0: "poisson (OOD)"}

FIELDS = ["group", "regime", "rho_label", "arm", "family", "seed", "content",
          "cost_mean", "cost_se", "cvar",
          "bw_retailer", "bw_wholesaler", "bw_distributor", "bw_manufacturer",
          "ready_rate", "holding_share",
          "V_vs_nocomm", "V_se", "cohen_dz", "V_ci_lo", "V_ci_hi", "holm_p",
          "paired_against",
          "V_vs_static", "V_vs_static_p", "gap_recovered", "gap_ci_lo", "gap_ci_hi",
          "listen_zeroed", "listen_shuffled", "fragility_excess",
          "signaling_r"]


def _g(d, *path, default=""):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def rows_from(path):
    d = json.load(open(path))
    group = os.path.splitext(os.path.basename(path))[0].replace("stats_", "")
    rho = d.get("rho", "")
    regime = _LABELS.get(float(rho), f"ar1 rho={rho}") if rho != "" else ""
    out = []
    for arm, blk in d.get("arms", {}).items():
        m = _RE_SEED.match(arm)
        pb = _g(d, "paired_vs_nocomm", arm, default={})
        vb = _g(blk, "vs_baselines", default={})
        li = _g(d, "listening", arm, default={})
        z = _g(li, "zeroed", "delta_cost_mean", default="")
        sh = _g(li, "shuffled", "delta_cost_mean", default="")
        bw = _g(blk, "bullwhip", "mean", default=[""] * 4)
        ci = _g(pb, "bca_95ci", default=["", ""])
        gci = _g(vb, "gap_recovered", "bca_95ci", default=["", ""])
        out.append({
            "group": group, "regime": regime, "rho_label": rho, "arm": arm,
            "family": m.group(1) if m else arm, "seed": m.group(2) if m else "",
            "content": blk.get("content", ""),
            "cost_mean": round(blk.get("cost_mean", float("nan")), 1),
            "cost_se": round(blk.get("cost_se", float("nan")), 1),
            "cvar": round(_g(blk, "cvar", "value", default=float("nan")), 1),
            "bw_retailer": round(bw[0], 2) if bw[0] != "" else "",
            "bw_wholesaler": round(bw[1], 2) if bw[1] != "" else "",
            "bw_distributor": round(bw[2], 2) if bw[2] != "" else "",
            "bw_manufacturer": round(bw[3], 2) if bw[3] != "" else "",
            "ready_rate": _g(blk, "service", "retailer_ready_rate", default=""),
            "holding_share": round(_g(blk, "decomposition", "holding_share",
                                      default=float("nan")), 3),
            "V_vs_nocomm": round(pb["V_mean"], 1) if pb else "",
            "V_se": round(pb["V_se"], 1) if pb else "",
            "cohen_dz": round(pb["cohen_dz"], 2) if pb else "",
            "V_ci_lo": round(ci[0], 1) if ci[0] != "" else "",
            "V_ci_hi": round(ci[1], 1) if ci[1] != "" else "",
            "holm_p": f"{pb['holm_p']:.2e}" if pb.get("holm_p") is not None else "",
            "paired_against": pb.get("paired_against", ""),
            "V_vs_static": round(_g(vb, "vs_static", "V_mean", default=float("nan")), 1)
                           if vb.get("available") else "",
            "V_vs_static_p": (f"{_g(vb, 'vs_static', 't_p', default=float('nan')):.2e}"
                              if vb.get("available") else ""),
            "gap_recovered": round(_g(vb, "gap_recovered", "mean",
                                      default=float("nan")), 3)
                             if vb.get("available") else "",
            "gap_ci_lo": round(gci[0], 3) if gci[0] != "" else "",
            "gap_ci_hi": round(gci[1], 3) if gci[1] != "" else "",
            "listen_zeroed": round(z, 1) if z != "" else "",
            "listen_shuffled": round(sh, 1) if sh != "" else "",
            "fragility_excess": round(z - sh, 1) if z != "" and sh != "" else "",
            "signaling_r": (round(_g(blk, "signaling", "pearson_r",
                                     default=float("nan")), 3)
                            if _g(blk, "signaling", "applicable", default=False) else ""),
        })
    aggs = []
    for fam, ag in d.get("seed_aggregate", {}).items():
        aggs.append({"group": group, "regime": regime, "family": fam,
                     "n_seeds": ag["n_seeds"], "V_mean": round(ag["V_seed_mean"], 1),
                     "between_seed_se": round(ag.get("V_between_seed_se",
                                                     float("nan")), 1),
                     "t_p": f"{ag.get('t_p', float('nan')):.3g}",
                     "sign_concordant": ag["sign_concordant"],
                     "per_seed": [round(v, 1) for v in ag["V_per_seed"]]})
    return out, aggs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default="runs/stats_*.json")
    ap.add_argument("--out", default="runs/RESULTS")
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(ROOT, a.stats)))
    if not paths:
        raise SystemExit(f"[collate] FAIL-CLOSED: no stats files match {a.stats}")

    rows, aggs = [], []
    for p in paths:
        try:
            r, g = rows_from(p)
            rows += r
            aggs += g
        except Exception as e:                       # never silently skip
            print(f"[collate] WARNING: {os.path.basename(p)} unreadable: {e}")

    csv_path = os.path.join(ROOT, a.out + ".csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    md = [f"# SIGNAL — consolidated results\n",
          f"Sources: {len(paths)} stats file(s); {len(rows)} arm-rows.\n"]
    md.append("\n## Seed-level aggregates (the headline numbers)\n")
    md.append("| group | regime | family | n | V mean | between-seed SE | p | "
              "concordant | per seed |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for g in aggs:
        md.append(f"| {g['group']} | {g['regime']} | {g['family']} | {g['n_seeds']} | "
                  f"{g['V_mean']} | {g['between_seed_se']} | {g['t_p']} | "
                  f"{'yes' if g['sign_concordant'] else '**NO**'} | {g['per_seed']} |")
    by_group = {}
    for r in rows:
        by_group.setdefault(r["group"], []).append(r)
    for grp, rs in by_group.items():
        md.append(f"\n## {grp}  ({rs[0]['regime']})\n")
        md.append("| arm | cost | V vs nocomm | dz | Holm p | V vs Static | "
                  "gap rec | listen(shuf) | fragility | signal r | bullwhip R/W/D/M |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(rs, key=lambda x: (x["family"], x["seed"])):
            bw = "/".join(str(r[f"bw_{e}"]) for e in
                          ("retailer", "wholesaler", "distributor", "manufacturer"))
            md.append(f"| {r['arm']} | {r['cost_mean']} | {r['V_vs_nocomm']} | "
                      f"{r['cohen_dz']} | {r['holm_p']} | {r['V_vs_static']} | "
                      f"{r['gap_recovered']} | {r['listen_shuffled']} | "
                      f"{r['fragility_excess']} | {r['signaling_r']} | {bw} |")
    md_path = os.path.join(ROOT, a.out + ".md")
    open(md_path, "w").write("\n".join(md) + "\n")

    print(f"[collate] {len(rows)} arm-rows from {len(paths)} stats file(s)")
    print(f"[collate] wrote {csv_path}")
    print(f"[collate] wrote {md_path}")
    disc = [g for g in aggs if not g["sign_concordant"]]
    if disc:
        print("[collate] NOTE: sign-DISCORDANT families (read these first):")
        for g in disc:
            print(f"           {g['group']}/{g['family']}  V {g['V_mean']} "
                  f"per-seed {g['per_seed']}")


if __name__ == "__main__":
    main()
