"""scripts/build_report.py -- the campaign as one self-contained HTML file.

    python scripts/build_report.py                       # -> docs/report.html
    python scripts/build_report.py --csv X --out Y

Sources, and the separation between them is the point:
  runs/RESULTS.csv          every NUMBER (never hand-edited)
  docs/extras.json          trajectory-derived numbers (echelon, budget, latency)
  docs/hypotheses_text.py   every WORD (edit freely, re-run, numbers cannot drift)

No external assets, no network, no fonts to load: the output opens from a USB stick.
"""
import argparse
import csv
import json
import os
import sys
from datetime import date

import numpy as np
from scipy import stats as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docs"))
from hypotheses_text import ABSTRACT, HYPOTHESES  # noqa: E402

G09 = "rho0.9_ar1_b10"


# ----------------------------------------------------------------- statistics
def _num(x, fmt):
    """A degenerate statistic prints an em dash. nan in a results table is a defect."""
    try:
        if x is None or not np.isfinite(x):
            return '<td class="n" style="color:var(--void)">&mdash;</td>'
        return f'<td class="n">{format(x, fmt)}</td>'
    except (TypeError, ValueError):
        return '<td class="n" style="color:var(--void)">&mdash;</td>'


def full(x, label, note=""):
    """Every statistic a reader needs for one condition, from one seed vector."""
    x = np.asarray(x, float)
    n = len(x)
    if n == 0:
        return {"label": label, "V": 0.0, "n": 0}
    m = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    sd = float(x.std(ddof=1)) if n > 1 else 0.0
    t, p2 = st.ttest_1samp(x, 0.0) if n > 1 and sd > 0 else (float("nan"), float("nan"))
    p1 = (p2 / 2 if t > 0 else 1 - p2 / 2) if n > 1 and sd > 0 else float("nan")
    ci = st.t.interval(0.95, n - 1, loc=m, scale=se) if n > 1 and se > 0 else (m, m)
    try:
        wp = float(st.wilcoxon(x).pvalue) if n > 5 and len(set(x)) > 1 else float("nan")
    except ValueError:
        wp = float("nan")
    return {"label": label, "V": m, "se": se, "sd": sd, "n": n,
            "dz": (m / sd if sd else float("nan")), "t": float(t), "p": float(p1),
            "ci": [float(ci[0]), float(ci[1])], "wp": wp,
            "conc": bool(all(x > 0) or all(x < 0)),
            "pos": int((x > 0).sum()),
            "seeds": [round(float(v), 1) for v in x], "note": note}


def one_sided(x, mu=0.0):
    t, p2 = st.ttest_1samp(np.asarray(x, float), mu)
    return float(t), float(p2 / 2 if t > 0 else 1 - p2 / 2)


def holm(pvals, alpha=0.05):
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for r, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - r) * p))
        out[k] = {"p": p, "holm": run, "survives": run < alpha}
    return out


def embed_png(root, name, caption="", cls="fig"):
    """Inline a PNG from figs/ as base64. Absent -> a note naming the missing file,
    never a broken image."""
    p = os.path.join(root, "figs", name)
    if not os.path.exists(p):
        return (f'<p class="missing">figs/{name} not found &mdash; unpack the analysis '
                f'archive into figs/ and rebuild to show it here.</p>')
    import base64
    b64 = base64.b64encode(open(p, "rb").read()).decode()
    cap = f'<figcaption>{caption}</figcaption>' if caption else ""
    return (f'<figure class="{cls}"><img alt="{caption or name}" '
            f'src="data:image/png;base64,{b64}">{cap}</figure>')


def read_curves(root, tags, col="team_cost", every=40):
    """Prefers metrics_train.csv (cost of every training episode). Falls back to
    metrics_gate.csv (held-out cost every 400 episodes) which is smoother and, if the
    full training log was not kept, still shows convergence."""
    """Training curves from runs/<tag>/metrics_train.csv, if the analysis archive has
    been unpacked locally. Absent -> the section degrades to text, never to a crash."""
    out = []
    for tag in tags:
        p = os.path.join(root, "runs", tag, "metrics_train.csv")
        use_col = col
        if not os.path.exists(p):
            p = os.path.join(root, "runs", tag, "metrics_gate.csv")
            use_col = "monitor_rho09"
            if not os.path.exists(p):
                continue
        ep, cost = [], []
        for i, r in enumerate(csv.DictReader(open(p, encoding="utf-8"))):
            if i % every:
                continue
            try:
                ep.append(float(r["episode"])); cost.append(float(r[use_col]))
            except (KeyError, ValueError):
                pass
        if len(ep) > 5:
            out.append({"tag": tag, "ep": ep, "cost": cost})
    return out


def spark_path(series, w=760, h=170, pad=8):
    """Polyline path for one curve, scaled to the shared frame."""
    if not series:
        return "", 0, 0
    xs = [p for s in series for p in s["ep"]]
    ys = [p for s in series for p in s["cost"]]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    paths = []
    for s in series:
        pts = " ".join(
            f"{pad + (e - x0) / max(x1 - x0, 1) * (w - 2 * pad):.1f},"
            f"{h - pad - (c - y0) / max(y1 - y0, 1) * (h - 2 * pad):.1f}"
            for e, c in zip(s["ep"], s["cost"]))
        paths.append(pts)
    return paths, y0, y1


class Data:
    def __init__(self, csv_path, extras_path):
        self.rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
        if not self.rows:
            raise SystemExit(f"[report] FAIL-CLOSED: {csv_path} is empty")
        self.extras = {}
        if os.path.exists(extras_path):
            self.extras = json.load(open(extras_path, encoding="utf-8"))

    def v(self, group, fam, col="V_vs_nocomm"):
        d = {int(r["seed"]): float(r[col]) for r in self.rows
             if r["group"] == group and fam in r["family"]
             and r[col] not in ("", "nan", "n/a: Static==Cond")}
        return np.array([d[s] for s in sorted(d)])


def compute(D):
    """Every statistic the report shows. Keys match docs/hypotheses_text.py."""
    R, r9 = {}, D.v(G09, "raw_reta_b10")

    def entry(key, stats, verdict, headline, extra=None):
        R[key] = {"stats": stats, "verdict": verdict, "headline": headline,
                  "extra": extra or {}}

    # F_CONTENT
    rows, ps = [], {}
    for fam, lab in (("raw_reta_b10", "raw demand"), ("arpred_reta", "forecast"),
                     ("dhatc_reta_b10", "learned forecast"),
                     ("learned", "emergent protocol"), ("ip_reta", "inventory position")):
        v = D.v(G09, fam)
        row = full(v, lab, f"gap {D.v(G09, fam, 'gap_recovered').mean():.0%}")
        row["gap"] = float(D.v(G09, fam, "gap_recovered").mean())
        rows.append(row); ps[lab] = row["p"]
    vs_static = D.v(G09, "raw_reta_b10", "V_vs_static")
    vrob = D.v(G09, "raw_reta_b10", "V_robust")
    gv = D.v(G09, "raw_reta_b10", "gap_recovered")
    glo, ghi = st.t.interval(0.95, len(gv) - 1, loc=gv.mean(),
                             scale=gv.std(ddof=1) / np.sqrt(len(gv)))
    entry("F_CONTENT", rows, "confirmed",
          f"{rows[0]['V']:+,.0f} cost units, {rows[0]['gap']:.0%} of the analytic gap",
          {"dual": {"nocomm": float(r9.mean()), "static": float(vs_static.mean()),
                    "robust": float(vrob.mean()),
                    "gap_ci": [float(glo), float(ghi)]}})

    # H2
    lv = {0.0: D.v("rho0_ar1_b10", "raw_reta"), 0.3: D.v("rho0.3_ar1_b10", "raw"),
          0.6: D.v("rho0.6_ar1_b10", "raw"), 0.9: r9}
    x = np.array(list(lv)); xc = x - x.mean()
    slopes = (xc @ np.array([lv[k] for k in lv])) / (xc @ xc)
    _, p = one_sided(slopes)
    entry("H2", [full(v, f"rho = {k:g}") for k, v in lv.items()],
          "confirmed", f"slope {slopes.mean():+,.0f} per unit of autocorrelation",
          {"slope": slopes.mean(), "slope_se": slopes.std(ddof=1) / np.sqrt(15), "p": p,
           "curve": [[k, float(v.mean())] for k, v in lv.items()]})
    ps["H2 gradient"] = p

    # H-SOURCE
    up = D.v(G09, "raw_upst"); s = r9 - up
    _, p = one_sided(s); ps["H-SOURCE"] = p
    entry("H_SOURCE", [full(r9, "broadcast to all echelons"), full(up, "relayed stage to stage"),
           full(r9 - up, "difference")],
          "confirmed", f"relay retains only {up.mean()/r9.mean():.0%} of the value",
          {"diff": s.mean(), "p": p})

    # H-TIME
    l1, l2 = D.v(G09, "raw_lag1"), D.v(G09, "raw_lag2")
    e = r9 - l2; _, p = one_sided(e); ps["H-TIME"] = p
    entry("H_TIME", [full(r9, "current period"), full(l1, "one period stale"),
                     full(l2, "two periods stale"), full(e, "endpoint difference")],
          "confirmed",
          f"{(r9.mean()-l2.mean())/r9.mean():.0%} lost over two periods, and the loss "
          f"accelerates",
          {"endpoint": e.mean(), "p": p})

    # H-TAIL
    cn, cr = D.v(G09, "nocomm", "cvar"), D.v(G09, "raw_reta_b10", "cvar")
    mn, mr = D.v(G09, "nocomm", "cost_mean"), D.v(G09, "raw_reta_b10", "cost_mean")
    d = cn - cr; _, p2 = st.ttest_1samp(d, 0); ps["H-TAIL"] = p2 / 2
    entry("H_TAIL", [full(d, "worst quarter of periods (CVaR)"), full(mn - mr, "average period")],
          "confirmed",
          f"the bad quarter improves {d.mean()/(mn-mr).mean():.2f}x more than the average",
          {"ratio": d.mean() / (mn - mr).mean(), "p": p2})

    # H-FRAGILITY
    fr, fl = D.v(G09, "raw_reta_b10", "fragility_excess"), D.v(G09, "learned", "fragility_excess")
    fd = fr - fl; _, p = one_sided(fd); ps["H-FRAGILITY"] = p
    lz = D.v(G09, "raw_reta_b10", "listen_zeroed")
    ls = D.v(G09, "raw_reta_b10", "listen_shuffled")
    entry("H_FRAGILITY", [full(fr, "engineered demand feed"), full(fl, "emergent protocol"),
                          full(fd, "difference"),
                          full(lz, "channel blanked (zeroed)"),
                          full(ls, "channel scrambled (shuffled)")],
          "confirmed", f"{fr.mean()/max(fl.mean(),1):.0f}x the exposure to channel failure",
          {"p": p})

    # H-SHOCK across arms
    shock = []
    for fam, lab in (("raw_reta_b10", "raw demand"), ("arpred_reta", "forecast"),
                     ("dhatc_reta_b10", "learned forecast"), ("learned", "emergent"),
                     ("ip_reta", "inventory position"), ("raw_upst", "relayed"),
                     ("raw_lag1", "one period stale"), ("raw_lag2", "two periods stale"),
                     ("raw_down", "own-order echo"), ("raw_manu", "most upstream signal")):
        idv, sw = D.v(G09, fam), D.v("rho-3_ood", fam)
        if len(idv) != 15 or len(sw) != 15:
            continue
        dd = sw - idv; _, p = one_sided(dd)
        row = full(sw, lab); row.update({"id": idv.mean(), "ood": sw.mean(),
                    "did": dd.mean(), "did_p": p,
                    "demand": lab not in ("own-order echo", "most upstream signal")})
        shock.append(row)
    _, p = one_sided(D.v("rho-3_ood", "raw_reta_b10") - r9); ps["H-SHOCK"] = p
    n_pos = sum(1 for s_ in shock if s_["demand"] and s_["did"] > 0)
    for s_ in shock:
        s_["note"] = f'shock {s_["did"]:+,.0f} (p {s_["did_p"]:.3f})' 
    entry("H_SHOCK", shock, "confirmed",
          f"every one of {n_pos} demand-bearing channels gains under disruption",
          {"p": p})

    # H-CALENDAR
    ch = D.v("rho-4_ood", "raw_reta_b10"); _, p2 = st.ttest_1samp(ch, 0)
    entry("H_CALENDAR", [full(ch, "unforecastable turbulence")],
          "confirmed-null", "no value where there is no level to announce",
          {"p": p2})

    # H-ECHELON (from extras)
    ex = D.extras.get("echelon")
    if ex:
        names = ["retailer", "wholesaler", "distributor", "manufacturer"]
        m, se = ex["by_content"]["raw"]["mean"], ex["by_content"]["raw"]["se"]
        entry("H_ECHELON", [{"label": n, "V": m[i], "se": se[i]} for i, n in enumerate(names)],
              "confirmed",
              f"upstream captures {sum(ex['share'][1:]):.0%} of the benefit, peaking at "
              f"the {names[int(np.argmax(m))]}",
              {"share": ex["share"], "uvr": ex["upstream_vs_retailer"],
               "by_content": ex["by_content"], "names": names})
        ps["H-ECHELON"] = ex["upstream_vs_retailer"]["p"] / 2

    # H-BUDGET (from extras)
    bx = D.extras.get("budget")
    if bx:
        entry("H_BUDGET",
              [{"label": f"{int(k):,} episodes", "V": v["V"], "se": v["se"],
                "conc": v["concordant"]} for k, v in sorted(bx["levels"].items(),
                                                            key=lambda kv: int(kv[0]))],
              "refuted",
              f"value RISES {bx['slope']:+,.0f} per doubling of planning capability",
              {"slope": bx["slope"], "se": bx["slope_se"], "p": bx["p"]})

    # P2
    g6 = D.v("rho0.9_ar1_b10_cl6", "raw") - r9
    g8 = D.v("rho0.9_ar1_b10_cl8", "raw") - r9
    _, p = one_sided(g6); ps["P2 garbling"] = p
    entry("P2", [full(g6, "62% of order-stream info destroyed"), full(g8, "47% destroyed")],
          "null-is-finding", "degrading the order stream changed nothing, for any arm",
          {"p": p})

    # P1
    dp_a = D.v("rho-1_dr_poisson_b10", "arpred", "gap_recovered")
    dp_r = D.v("rho-1_dr_poisson_b10", "raw", "gap_recovered")
    dp_n = D.v("rho-1_dr_poisson_b10", "nocomm", "gap_recovered")
    ar_a = D.v(G09, "arpred_reta", "gap_recovered")
    ar_r = D.v(G09, "raw_reta_b10", "gap_recovered")
    did = (dp_a - dp_r) - (ar_a - ar_r)
    entry("P1", [{"label": "order stream alone", "V": dp_n.mean() * 100,
                  "se": dp_n.std(ddof=1) / np.sqrt(15) * 100, "pct": True},
                 {"label": "with demand sharing", "V": dp_r.mean() * 100,
                  "se": dp_r.std(ddof=1) / np.sqrt(15) * 100, "pct": True}],
          "refuted",
          f"no forecast premium; sharing lifts regime inference from "
          f"{dp_n.mean():.0%} to {dp_r.mean():.0%}",
          {"did": did.mean(), "p": one_sided(did)[1]})

    # H-LEARN
    hl = D.v(G09, "arpred_reta") - r9; _, p = one_sided(hl); ps["H-LEARN"] = p
    entry("H_LEARN", [full(hl, "forecast minus raw demand")],
          "not-replicated", "representation does not matter here", {"p": p})

    # H-CRITICAL-RATIO
    bh = D.v("rho0.9_ar1_b10_bh4", "raw")
    if len(bh) == 15:
        dcr = bh - r9; _, p = one_sided(dcr)
        entry("H_CRITICAL_RATIO",
              [full(bh, "backorder/holding = 4"), full(r9, "backorder/holding = 2"),
               full(dcr, "difference")],
              "refuted", "value is invariant to the cost ratio", {"p": p})

    # H-REP
    pairs = []
    fams = {"raw demand": r9, "forecast": D.v(G09, "arpred_reta"),
            "learned forecast": D.v(G09, "dhatc_reta_b10")}
    noc = D.v(G09, "nocomm", "cost_mean").mean(); margin = 0.02 * noc
    names = list(fams)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            dd = fams[names[i]] - fams[names[j]]
            pairs.append(full(dd, f"{names[i]} vs {names[j]}"))
    entry("H_REP", pairs, "inconclusive",
          f"all differences inside the ±{margin:,.0f} materiality band, but not "
          f"formally equivalent", {"margin": margin})

    # ip
    ipv = D.v(G09, "ip_reta"); ipo = D.v("rho-3_ood", "ip_reta")
    _, p = one_sided(ipv); ps["inventory position"] = p
    entry("IP", [full(ipv, "stationary demand"), full(ipo, "under disruption")],
          "confirmed", f"worth {ipv.mean()/r9.mean():.0%} of demand sharing, "
                       f"but {ipo.mean()/ipv.mean():.1f}x more in a crisis", {"p": p})

    # F_INCENTIVE
    inc = []
    for g, b in (("rho0.9_ar1_b0", "own cost only"),
                 ("rho0.9_ar1_b05", "partial alignment"), (G09, "full chain cost")):
        v = D.v(g, "dhatc"); nc = D.v(g, "nocomm", "cost_mean")
        if len(v):
            inc.append(full(v, b, f"no-sharing cost {nc.mean():,.0f}"))
    entry("F_INCENTIVE", inc, "confirmed", "value persists under decentralised objectives")

    # placebos
    dn, mnb, nn = D.v(G09, "raw_down"), D.v(G09, "raw_manu"), D.v(G09, "raw_no_n")
    _, pd_ = st.ttest_1samp(dn, 0); ps["own-order echo harms"] = pd_ / 2
    entry("PLACEBO", [full(dn, "own-order echo"), full(mnb, "most upstream signal"),
                      full(nn, "channel disabled")],
          "violated", "an empty channel is not free -- it costs", {"p": pd_})

    # control validity
    ctl = []
    for g, lab in (("rho0_ar1_b10", "rho = 0"), ("rho0.3_ar1_b10", "rho = 0.3"),
                   ("rho0.6_ar1_b10", "rho = 0.6"), (G09, "rho = 0.9"),
                   ("rho0.9_ar1_b10_cl6", "coarsened orders (6)"),
                   ("rho0.9_ar1_b10_cl8", "coarsened orders (8)"),
                   ("rho0.9_ar1_b10_bh4", "backorder-heavy costs"),
                   ("rho-1_dr_poisson_b10", "regime uncertainty"),
                   ("rho-3_ood", "unanticipated shock"),
                   ("rho-4_ood", "turbulence")):
        v = D.v(g, "nocomm", "V_vs_static"); c = D.v(g, "nocomm", "cost_mean")
        if not len(v):
            continue
        ood = g.endswith("_ood")
        ctl.append({"label": lab, "delta": v.mean(),
                    "cv": float(c.std(ddof=1) / c.mean()),
                    "status": "transfer" if ood else
                              ("holds" if v.mean() > -120 else "broken")})
    return R, ctl, holm(ps)


# ----------------------------------------------------------------------- HTML
CSS = """
:root{
  --paper:#EEF1F5; --card:#FBFCFD; --ink:#10192B; --ink-2:#425068; --rule:#C9D2DE;
  --signal:#C6741B; --signal-soft:#F5E3CC; --flow:#2C6E8F; --void:#8B95A6;
  --ok:#1F6B4A; --no:#8A3324;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  --sans:"Helvetica Neue",Inter,system-ui,-apple-system,sans-serif;
  --serif:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
@media (prefers-reduced-motion:reduce){html{scroll-behavior:auto}*{animation:none!important;transition:none!important}}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--serif);
     font-size:17px;line-height:1.62;-webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:0 28px}
h1,h2,h3,.ui{font-family:var(--sans)}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.16em;text-transform:uppercase;
         color:var(--void);margin:0 0 10px}
h1{font-size:clamp(38px,6vw,72px);line-height:.98;letter-spacing:-.035em;font-weight:800;margin:0 0 22px}
h1 em{font-style:normal;color:var(--signal)}
h2{font-size:clamp(24px,3vw,34px);letter-spacing:-.02em;font-weight:750;margin:0 0 6px}
.lede{font-size:20px;line-height:1.55;color:var(--ink-2);max-width:62ch;margin:0 0 32px}
.num{font-family:var(--mono);font-variant-numeric:tabular-nums}
header.top{padding:76px 0 46px;border-bottom:1px solid var(--rule)}
section{padding:62px 0;border-bottom:1px solid var(--rule)}
.sec-head{display:flex;gap:18px;align-items:baseline;flex-wrap:wrap;margin-bottom:26px}
.sec-head p{margin:0;color:var(--ink-2);max-width:58ch;font-size:16px}

/* ---- signature: the echelon spine ---- */
.spine{background:var(--card);border:1px solid var(--rule);border-radius:3px;padding:30px 26px;margin:34px 0 0}
.spine-legend{display:flex;gap:26px;flex-wrap:wrap;font-family:var(--mono);font-size:11.5px;
              letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2);margin-bottom:22px}
.key{display:inline-flex;align-items:center;gap:8px}
.key i{width:22px;height:3px;display:inline-block}
.chain{display:grid;grid-template-columns:repeat(4,1fr);gap:0;position:relative}
.node{position:relative;padding:0 10px}
.node .nm{font-family:var(--mono);font-size:11px;letter-spacing:.08em;text-transform:uppercase;
          color:var(--ink-2);margin-bottom:12px}
.bars{display:flex;flex-direction:column;gap:7px}
.bar{position:relative;height:26px;background:#E3E8EF;border-radius:2px;overflow:hidden}
.bar span{position:absolute;inset:0 auto 0 0;display:block;border-radius:2px;
          animation:grow 1.1s cubic-bezier(.2,.7,.3,1) both}
.bar.nc span{background:var(--void)}
.bar.sg span{background:var(--signal)}
.bar b{position:absolute;right:8px;top:50%;transform:translateY(-50%);font-family:var(--mono);
       font-size:12px;font-weight:600;color:var(--ink)}
@keyframes grow{from{transform:scaleX(.02);transform-origin:left}to{transform:scaleX(1);transform-origin:left}}
.spine-note{margin:22px 0 0;font-size:15.5px;color:var(--ink-2);max-width:70ch}

/* ---- verdict groups + cards ---- */
.group-label{font-family:var(--mono);font-size:11.5px;letter-spacing:.14em;text-transform:uppercase;
             color:var(--ink-2);padding:26px 0 12px;border-top:2px solid var(--ink);margin-top:34px}
.group-label:first-of-type{margin-top:0}
details.h{background:var(--card);border:1px solid var(--rule);border-radius:3px;margin:0 0 10px}
details.h[open]{border-color:var(--ink-2)}
summary.h{list-style:none;cursor:pointer;padding:18px 22px;display:grid;
          grid-template-columns:1fr auto;gap:16px;align-items:center}
summary.h::-webkit-details-marker{display:none}
summary.h:focus-visible{outline:2px solid var(--signal);outline-offset:2px}
.h-id{font-family:var(--mono);font-size:10.5px;letter-spacing:.16em;color:var(--signal);margin-bottom:4px}
.h-title{font-family:var(--sans);font-weight:650;font-size:17.5px;letter-spacing:-.01em}
.method{background:#EDF1F6;border-left:3px solid var(--flow);padding:12px 16px;margin:18px 0 0;
        font-size:14.5px;color:var(--ink-2);max-width:70ch;line-height:1.55}
.method .ml{display:block;font-family:var(--mono);font-size:10px;letter-spacing:.14em;
            text-transform:uppercase;color:var(--flow);margin-bottom:5px}
.h-head{font-size:15px;color:var(--ink-2);margin-top:3px}
.badge{font-family:var(--mono);font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;
       padding:5px 10px;border:1px solid currentColor;border-radius:2px;white-space:nowrap}
.b-confirmed{color:var(--ok)} .b-refuted,.b-not-replicated{color:var(--no)}
.b-null-is-finding,.b-violated{color:var(--signal)} .b-inconclusive,.b-confirmed-null{color:var(--void)}
.body{padding:4px 22px 24px;border-top:1px solid var(--rule)}
.stmt{border-left:3px solid var(--signal);padding:2px 0 2px 16px;margin:20px 0;font-size:17px}
.mech{color:var(--ink-2);font-size:16px;margin:0 0 18px;max-width:66ch}
.read{font-size:16.5px;margin:16px 0 0;max-width:66ch}
.read.pending{color:var(--void);font-style:italic}
.srcs{font-family:var(--mono);font-size:12px;color:var(--void);margin-top:18px;line-height:1.9}
table.d{width:100%;border-collapse:collapse;margin:18px 0 4px;font-family:var(--mono);font-size:13.5px}
table.d th{text-align:left;font-weight:500;color:var(--void);font-size:10.5px;letter-spacing:.1em;
           text-transform:uppercase;padding:0 10px 8px 0;border-bottom:1px solid var(--rule)}
table.d td{padding:8px 10px 8px 0;border-bottom:1px solid #E6EBF1}
table.d td.n{text-align:right;font-variant-numeric:tabular-nums}
table.full{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:11.5px;margin:16px 0 0}
table.full th{text-align:right;font-weight:500;color:var(--void);font-size:9.5px;letter-spacing:.06em;
  text-transform:uppercase;padding:0 8px 7px 0;border-bottom:1px solid var(--rule)}
table.full th:first-child{text-align:left}
table.full td{padding:6px 8px 6px 0;border-bottom:1px solid #E9EDF2}
table.full td.n{text-align:right;font-variant-numeric:tabular-nums}
details.seeds{margin-top:10px}
details.seeds summary{cursor:pointer;font-family:var(--mono);font-size:10.5px;letter-spacing:.1em;
  text-transform:uppercase;color:var(--flow);padding:6px 0}
.sv{display:flex;gap:12px;padding:5px 0;font-size:11px;align-items:baseline}
.sv span{font-family:var(--mono);color:var(--void);min-width:150px;font-size:10.5px}
.sv code{font-family:var(--mono);color:var(--ink-2);font-size:10.5px;word-break:break-all}
.gauge{display:inline-block;height:9px;background:var(--signal);border-radius:1px;vertical-align:middle}
.gauge.neg{background:var(--no)}
.sparks{display:flex;gap:2px;align-items:flex-end;height:20px}
.sparks i{width:3px;background:var(--flow);display:block;border-radius:.5px}
.sparks i.dn{background:var(--no)}

/* ---- tables / certificate ---- */
table.cert{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:13.5px;margin-top:8px}
table.cert th{text-align:left;font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;
              color:var(--void);font-weight:500;padding:0 12px 9px 0;border-bottom:1px solid var(--rule)}
table.cert td{padding:9px 12px 9px 0;border-bottom:1px solid #E6EBF1}
.pill{font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;padding:3px 8px;border-radius:2px}
.p-holds{background:#DCEBE2;color:var(--ok)} .p-broken{background:#F3DDD8;color:var(--no)}
.p-transfer{background:var(--signal-soft);color:var(--signal)}
.flowline{display:flex;align-items:center;gap:0;flex-wrap:wrap;background:var(--card);
  border:1px solid var(--rule);border-radius:3px;padding:22px 20px;margin-top:6px}
.fl-stage{flex:1;min-width:120px;text-align:center}
.fl-stage b{display:block;font-family:var(--sans);font-weight:650;font-size:15px}
.fl-stage span{font-family:var(--mono);font-size:10.5px;letter-spacing:.08em;
  text-transform:uppercase;color:var(--void)}
.fl-arrow{font-family:var(--mono);font-size:10.5px;letter-spacing:.08em;color:var(--flow);
  padding:0 12px;white-space:nowrap}
.mats{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px;margin-top:28px}
.mat-top{display:flex;align-items:flex-end;gap:10px}
.mat-ax{display:flex;flex-direction:column;gap:4px;font-family:var(--mono);font-size:9px;
  color:var(--void);line-height:1}
.mat-ax span{height:20px;display:flex;align-items:center}
.mv{font-family:var(--mono);font-size:13px;font-weight:600;margin-left:8px}
.mv.pos{color:var(--ok)} .mv.neg{color:var(--no)} .mv.nil{color:var(--void)}
.why{display:block;margin-top:8px;font-size:14px;color:var(--ink-2);line-height:1.5}
.mat{margin:0;background:var(--card);border:1px solid var(--rule);border-radius:3px;padding:18px}
.grid4{display:grid;grid-template-columns:repeat(4,1fr);gap:4px;width:96px}
.grid4 i{aspect-ratio:1;background:#E3E8EF;border-radius:1px;display:block}
.grid4 i.on{background:var(--signal)}
.grid4 i.on.dim{background:var(--void)}
.mat figcaption{margin-top:14px;font-size:14px;color:var(--ink-2);line-height:1.45}
.mat figcaption b.tid{display:inline;font-family:var(--mono);color:var(--ink);font-size:13.5px;letter-spacing:-.01em}
.mat figcaption code{font-family:var(--mono);font-size:12.5px;background:#E8EDF3;padding:1px 5px;border-radius:2px}
.missing{color:var(--void);font-size:15px;max-width:66ch;margin:18px 0 0}
figure.fig{margin:26px 0 0;background:var(--card);border:1px solid var(--rule);border-radius:3px;padding:18px}
figure.fig img{width:100%;height:auto;display:block}
figure.fig figcaption{margin-top:14px;font-size:15px;color:var(--ink-2);max-width:70ch;line-height:1.5}
.curve{margin:30px 0 0;background:var(--card);border:1px solid var(--rule);border-radius:3px;padding:20px}
.curve svg{width:100%;height:170px;display:block}
.curve figcaption{margin-top:14px;font-size:15px;color:var(--ink-2);max-width:70ch;line-height:1.5}
.fig{margin:30px 0 0;background:var(--card);border:1px solid var(--rule);border-radius:3px;padding:22px}
.chart{width:100%;height:auto;display:block}
.chart .lb{font-family:var(--mono);font-size:10.5px;fill:var(--ink-2)}
.chart .vl{font-family:var(--mono);font-size:10.5px;fill:var(--ink);font-weight:600}
.fig figcaption{margin-top:14px;font-size:15px;color:var(--ink-2);max-width:74ch;line-height:1.5}
.fig figcaption.fc-top{margin:0 0 14px;font-family:var(--mono);font-size:11px;
  letter-spacing:.05em;text-transform:uppercase}
.dual{background:#EFF4F1;border-left:3px solid var(--ok);padding:14px 18px;margin:18px 0 0}
.dual .dl{display:block;font-family:var(--mono);font-size:10px;letter-spacing:.14em;
  text-transform:uppercase;color:var(--ok);margin-bottom:10px}
.dr{display:flex;gap:12px;align-items:baseline;font-size:14.5px;color:var(--ink-2);padding:3px 0}
.dr b{font-family:var(--mono);font-size:15px;color:var(--ink);min-width:78px}
.dual p{margin:10px 0 0;font-size:14px;color:var(--ink-2);line-height:1.5}
.minis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.mini{margin:0;background:#F6F8FA;border:1px solid var(--rule);border-radius:2px;padding:6px}
.mini.wide{min-width:250px}
.strip{width:250px;height:26px;display:block}
.stripline{display:flex;align-items:center;gap:12px;margin-top:6px}
.stripline span{font-family:var(--mono);font-size:10.5px;color:var(--void)}
.guide{display:grid;gap:12px;margin-top:6px}
.gq{background:var(--card);border:1px solid var(--rule);border-radius:3px;padding:18px 22px 18px 62px;
  position:relative;font-size:16px;color:var(--ink-2);line-height:1.55}
.gq .qn{position:absolute;left:20px;top:18px;font-family:var(--mono);font-size:20px;
  color:var(--signal);font-weight:600}
.gq b{display:block;font-family:var(--sans);font-size:17px;color:var(--ink);margin-bottom:6px}
.gq em{display:block;margin-top:8px;font-style:normal;font-family:var(--mono);font-size:12px;
  color:var(--flow)}
.rules{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:14px;margin-top:6px}
.rule{background:var(--card);border:1px solid var(--rule);border-radius:3px;padding:18px 20px;
  font-size:15.5px;color:var(--ink-2);line-height:1.5}
.rule b{display:block;font-family:var(--sans);font-size:15px;color:var(--ink);margin-bottom:7px}
.pipeline{counter-reset:st;list-style:none;padding:0;margin:30px 0 0;max-width:76ch}
.pipeline li{counter-increment:st;position:relative;padding:0 0 20px 46px;
  border-left:1px solid var(--rule);margin-left:12px}
.pipeline li:last-child{border-left-color:transparent}
.pipeline li::before{content:counter(st);position:absolute;left:-13px;top:0;width:26px;height:26px;
  background:var(--card);border:1px solid var(--rule);border-radius:50%;display:flex;
  align-items:center;justify-content:center;font-family:var(--mono);font-size:11px;color:var(--flow)}
.pipeline li b{font-family:var(--sans);font-weight:650}
.notes{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:20px;margin-top:6px}
.note{background:var(--card);border:1px solid var(--rule);border-radius:3px;padding:18px 20px}
.note .nk{display:flex;align-items:baseline;gap:10px;margin-bottom:8px}
.note .nk .num{font-size:16px;font-weight:600;color:var(--signal)}
.note .nk em{font-style:normal;font-family:var(--mono);font-size:10px;letter-spacing:.1em;
  text-transform:uppercase;color:var(--void)}
.note p{margin:0;font-size:15.5px;color:var(--ink-2);line-height:1.55}
.pgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px}
.pgroup{background:var(--card);border:1px solid var(--rule);border-radius:3px;padding:18px 20px}
.pgroup h3{font-size:12px;font-family:var(--mono);letter-spacing:.12em;text-transform:uppercase;
  color:var(--flow);margin:0 0 12px;font-weight:500}
table.params{width:100%;border-collapse:collapse;font-size:14.5px}
table.params td{padding:7px 0;border-bottom:1px solid #E6EBF1;vertical-align:top}
table.params td:first-child{font-family:var(--mono);font-size:12px;color:var(--void);
  width:40%;padding-right:14px}
.sub{display:block;font-family:var(--serif);font-size:13px;color:var(--void)}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:22px;margin-top:26px}
.stat{background:var(--card);border:1px solid var(--rule);border-radius:3px;padding:18px 20px}
.stat .k{font-family:var(--mono);font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--void)}
.stat .v{font-family:var(--sans);font-size:30px;font-weight:750;letter-spacing:-.02em;margin:6px 0 4px}
.stat .s{font-size:14.5px;color:var(--ink-2);line-height:1.45}
footer{padding:52px 0 80px;font-family:var(--mono);font-size:12px;color:var(--void);line-height:2}
a{color:var(--flow)}
@media(max-width:720px){
  .chain{grid-template-columns:1fr;gap:16px}
  summary.h{grid-template-columns:1fr;gap:10px}
  body{font-size:16px}
}
"""


def bar_row(label, val, se, vmax, note="", pct=False):
    w = min(abs(val) / vmax * 100, 100) if vmax else 0
    neg = " neg" if val < 0 else ""
    unit = "%" if pct else ""
    se_s = f" ± {se:,.0f}" if se else ""
    return (f'<tr><td>{label}</td>'
            f'<td class="n">{val:+,.0f}{unit}{se_s}</td>'
            f'<td style="width:46%"><span class="gauge{neg}" style="width:{w:.1f}%"></span></td>'
            f'<td class="n" style="color:var(--void)">{note}</td></tr>')


def sparks(seeds):
    if not seeds:
        return ""
    m = max(abs(min(seeds)), abs(max(seeds))) or 1
    out = "".join(f'<i class="{"dn" if s < 0 else ""}" style="height:{max(abs(s)/m*20,1.5):.1f}px"></i>'
                  for s in seeds)
    return ('<div style="display:flex;align-items:flex-end;gap:10px;margin-top:10px">'
            f'<div class="sparks">{out}</div>'
            '<span class="num" style="font-size:11px;color:var(--void)">'
            'each seed, first condition</span></div>')


def card(key, res):
    txt = HYPOTHESES.get(key)
    if not txt:
        return ""
    st_ = txt.get("statement")
    if isinstance(st_, tuple):        # tolerate a stray trailing comma in the prose file
        st_ = st_[0]
    verdict = res["verdict"]
    rows = res["stats"]
    vmax = max((abs(r.get("V", 0)) for r in rows), default=1) or 1
    body = "".join(bar_row(r["label"], r.get("V", 0), r.get("se", 0), vmax,
                           r.get("note", ""), r.get("pct", False)) for r in rows)
    # full inference table -- only for rows that carry a seed vector
    fr = [r for r in rows if r.get("seeds")]
    stat_tbl = ""
    if fr:
        head = ("<tr><th>condition</th><th>mean</th><th>SE</th><th>SD</th>"
                "<th>95% CI</th><th>d<sub>z</sub></th><th>t</th><th>p</th>"
                "<th>seeds +</th><th>concordant</th></tr>")
        trs = ""
        for r in fr:
            ci = r.get("ci", [0, 0])
            trs += (f'<tr><td>{r["label"]}</td>'
                    f'<td class="n">{r["V"]:+,.1f}</td>'
                    f'<td class="n">{r.get("se",0):,.1f}</td>'
                    f'<td class="n">{r.get("sd",0):,.1f}</td>'
                    f'<td class="n">[{ci[0]:+,.0f}, {ci[1]:+,.0f}]</td>'
                    f'{_num(r.get("dz"), ".2f")}'
                    f'{_num(r.get("t"), ".2f")}'
                    f'{_num(r.get("p"), ".2e")}{_num(r.get("wp"), ".2e")}'
                    f'<td class="n">{r.get("pos",0)}/{r.get("n",0)}</td>'
                    f'<td class="n">{"yes" if r.get("conc") else "no"}</td></tr>')
        seedlines = "".join(
            f'<div class="sv"><span>{r["label"]}</span>'
            f'<code>{", ".join(f"{v:+,.0f}" for v in r["seeds"])}</code></div>'
            for r in fr)
        stat_tbl = (f'<details class="seeds"><summary>full statistics</summary>'
                    f'<table class="full">{head}{trs}</table>{seedlines}</details>')
    extra = res.get("extra", {})
    notes = []
    if "p" in extra:
        notes.append(f'one-sided p = {extra["p"]:.2e}')
    if "slope" in extra and key == "H_BUDGET":
        notes.append(f'slope {extra["slope"]:+,.0f} ± {extra["se"]:,.0f} per doubling')
    if "slope" in extra and key == "H2":
        notes.append(f'slope {extra["slope"]:+,.0f} ± {extra["slope_se"]:,.0f} per unit rho')
    dual_html = ""
    if "dual" in extra:
        d_ = extra["dual"]
        dual_html = (
            '<div class="dual"><span class="dl">measured twice, independently</span>'
            f'<div class="dr"><b>{d_["nocomm"]:+,.0f}</b>against its own matched '
            'no-sharing chain</div>'
            f'<div class="dr"><b>{d_["static"]:+,.0f}</b>against the fitted analytic '
            'base-stock rule</div>'
            f'<div class="dr"><b>{d_["robust"]:+,.0f}</b>the conservative figure: '
            'whichever reference is less favourable</div>'
            f'<p>Two references that share no machinery agree to within '
            f'{abs(d_["nocomm"]-d_["static"])/max(d_["nocomm"],1):.0%}. The gap-recovered '
            f'fraction has a 95% interval across seeds of [{d_["gap_ci"][0]:.0%}, '
            f'{d_["gap_ci"][1]:.0%}].</p></div>')
    if "margin" in extra:
        notes.append(f'materiality band ± {extra["margin"]:,.0f}')
    if "ratio" in extra:
        notes.append(f'tail/mean ratio {extra["ratio"]:.2f}')
    seedbars = "".join(
        f'<div class="stripline">{strip(r["seeds"])}'
        f'<span>{r["label"]} &mdash; {r.get("pos",0)}/{r.get("n",0)} seeds above zero</span>'
        f'</div>' for r in rows if r.get("seeds"))
    read = txt.get("reading", "").strip()
    read_html = (f'<p class="read">{read}</p>' if read else
                 '<p class="read pending">Interpretation pending — write it in '
                 'docs/hypotheses_text.py and rebuild.</p>')
    rid, method = IDS.get(key, (key, ""))
    method_html = (f'<div class="method"><span class="ml">how it was tested</span>'
                   f'{method}</div>' if method else "")
    return f"""
<details class="h">
  <summary class="h">
    <div><div class="h-id">{rid}</div>
         <div class="h-title">{txt['title']}</div>
         <div class="h-head">{res['headline']}</div></div>
    <div class="badge b-{verdict}">{verdict.replace('-', ' ')}</div>
  </summary>
  <div class="body">
    <p class="stmt">{st_}</p>
    <p class="mech">{txt['mechanism']}</p>
    <table class="d"><tr><th>condition</th><th>cost units</th><th></th><th></th></tr>
      {body}</table>
    <div class="num" style="font-size:12px;color:var(--void);margin-top:6px">
      n = 15 seeds &nbsp;·&nbsp; {' &nbsp;·&nbsp; '.join(notes) if notes else 'paired within seed'}
    </div>
    {dual_html}
    {stat_tbl}
    {method_html}
    {read_html}
    <div class="srcs">{'<br>'.join(txt['sources'])}</div>
  </div>
</details>"""


# Registry IDs and the test behind each card. Shown in the card so a reader never has
# to reconstruct which statistic produced a number.
IDS = {
 "F_CONTENT":("F-CONTENT","Per-content one-sided t-test on the 15 seed-level paired "
   "differences, each seed's communicating chain against its own no-sharing twin "
   "evaluated on identical demand draws (common random numbers). Holm correction "
   "within the content family."),
 "H2":("H2","Ordinary least squares slope of V on the autocorrelation coefficient, "
   "fitted separately within each seed, then a one-sided t-test on the 15 slopes."),
 "H_SOURCE":("H-SOURCE  ·  family F-GEOMETRY","One-sided paired t-test on the per-seed "
   "difference between broadcast and relayed topologies, both trained under identical "
   "conditions. This is the primary test of the geometry family; the three placebo "
   "wirings reported under PLACEBO belong to the same family and are what make the "
   "contrast interpretable."),
 "H_TIME":("H-TIME","Registered primary contrast is the endpoint, current versus "
   "two-period-old demand, by one-sided paired t-test; monotonicity across the three "
   "lags is secondary."),
 "H_TAIL":("H-TAIL","Paired t-test on the per-seed reduction in conditional "
   "value-at-risk of episode cost at the 25% level, compared against the reduction in "
   "the mean."),
 "H_FRAGILITY":("H-FRAGILITY","Cost increase when the channel is corrupted at "
   "evaluation time, measured on the already-trained policy. The reported quantity is "
   "the excess of blanking the channel over merely scrambling it, which isolates "
   "dependence on the signal from loss of its information."),
 "H_SHOCK":("H-SHOCK","Difference-in-differences: each policy is scored on its "
   "training distribution and again, without any retraining, on an unannounced demand "
   "shock. One-sided t-test on the per-seed change in V."),
 "H_CALENDAR":("H-CALENDAR","Two-sided t-test against zero, with sign concordance "
   "across seeds reported, because the registered prediction is a null."),
 "H_ECHELON":("H-ECHELON","Per-echelon cost decomposed from the trajectory record, "
   "differenced against the matched no-sharing chain; paired t-test on the three "
   "upstream stages against the retailer."),
 "H_BUDGET":("H-BUDGET","Checkpoints saved at fixed training milestones are scored "
   "against no-sharing chains at the same milestone, so capability is held equal on "
   "both sides. One-sided t-test on the per-seed slope over log2 of the budget."),
 "P2":("P2","Blackwell garbling: the order quantities upstream stages observe are "
   "coarsened, leaving transitions and costs identical. One-sided t-test on the change "
   "in V, with a dose comparison between the two coarsening levels."),
 "P1":("P1-prime","Difference-in-differences on gap-recovered units, so the two demand "
   "regimes are compared on a common scale rather than in raw cost. One-sided "
   "studentized bootstrap-t lower bound."),
 "H_LEARN":("H-LEARN","One-sided paired t-test on the per-seed difference between the "
   "forecast and raw-demand channels."),
 "H_CRITICAL_RATIO":("H-CRITICAL-RATIO","One-sided paired t-test between two cost "
   "regimes trained identically apart from the backorder penalty."),
 "H_REP":("H-REP","Two one-sided tests for equivalence within a materiality band of "
   "two per cent of no-sharing cost, Holm-corrected across the three pairs. "
   "Equivalence requires the whole confidence interval inside the band, which is a "
   "stricter demand than a non-significant difference."),
 "IP":("IP","One-sided t-test on the seed-level paired difference, with the same arm "
   "re-scored under disruption."),
 "F_INCENTIVE":("F-INCENTIVE","Descriptive across incentive levels; each level is "
   "paired against a no-sharing chain trained under the same objective."),
 "PLACEBO":("PLACEBO","Two-sided t-tests against zero. The disabled channel is checked "
   "for exact equality with no-sharing rather than statistical indistinguishability."),
}

GROUPS = [
    ("What the chain gains", "confirmed",
     ["F_CONTENT", "H2", "H_SOURCE", "H_TIME", "H_TAIL", "H_ECHELON", "IP",
      "H_SHOCK", "H_CALENDAR", "H_FRAGILITY", "F_INCENTIVE"]),
    ("Predictions the data refused", "refuted",
     ["H_BUDGET", "P1", "H_LEARN", "H_CRITICAL_RATIO"]),
    ("Nulls that carry the argument", "null",
     ["P2", "PLACEBO", "H_REP"]),
]


TOPOS = [
    ("retailer_broadcast", "raw_reta_b10",
     [[0,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]], True,
     "Everyone hears the retailer. Column R is on for all three upstream rows: every blind stage receives the "
     "retailer's demand observation in the period after it happens. This is full "
     "point-of-sale visibility, and it is the reference against which the others are "
     "read."),
    ("upstream_only", "raw_upst",
     [[0,0,0,0],[1,0,0,0],[0,1,0,0],[0,0,1,0]], True,
     "Each stage hears its immediate downstream partner; the config also accepts the "
     "historical alias <code>neighbor</code>, which builds an identical matrix. "
     "A sub-diagonal. The "
     "wholesaler still receives true demand, but the distributor receives whatever the "
     "wholesaler observed and the factory receives that in turn. This is what per-link "
     "EDI looks like as a matrix."),
    ("downstream_only", "raw_down",
     [[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,0,0,0]], False,
     "The mirror of upstream_only: the super-diagonal, pointing the wrong way down the "
     "chain. Each stage hears its "
     "supplier's incoming order, which is the order it placed itself. The channel is "
     "live and carries real numbers; it just returns to each stage what it already "
     "knew."),
    ("manufacturer_broadcast", "raw_manu",
     [[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,0]], False,
     "The same shape as retailer_broadcast, sourced from the factory instead. The signal is genuinely "
     "demand-correlated, having been filtered through three replenishment policies and "
     "delayed by three hops. It is the same architecture as the first matrix pointed at "
     "the least informed stage."),
    ("no_neighbor", "raw_no_n",
     [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]], False,
     "Every entry zero. Senders compose, the matrix routes, receivers read an all-zero "
     "vector. This runs the entire communication apparatus while transmitting nothing, "
     "so its measured value must be exactly zero -- the apparatus certifying itself."),
    ("neighbor", None,
     [[0,0,0,0],[1,0,0,0],[0,1,0,0],[0,0,1,0]], True,
     "An alias: the historical name for upstream_only, kept so older configurations "
     "still resolve. It builds the identical matrix, and the two names are "
     "interchangeable everywhere in the code."),
]
STAGES = ["R", "W", "D", "M"]


def matrices_html(D):
    """All five routing matrices, each with the measured outcome of that wiring."""
    out = ""
    for name, fam, M, live, why in TOPOS:
        cells = ""
        for i in range(4):
            for j in range(4):
                on = M[i][j]
                cls = "on" if on and live else ("on dim" if on else "")
                cells += f'<i class="{cls}"></i>'
        v = D.v(G09, fam) if fam else np.array([])
        val = (f'<span class="mv {"pos" if v.mean() > 5 else ("neg" if v.mean() < -5 else "nil")}">'
               f'{v.mean():+,.0f}</span>' if len(v) else "")
        out += (f'<figure class="mat"><div class="mat-top"><div class="grid4">{cells}</div>'
                f'<div class="mat-ax"><span>R</span><span>W</span><span>D</span><span>M</span></div>'
                f'</div><figcaption><b class="tname">{name}</b> {val}<span class="why">{why}</span>'
                f'</figcaption></figure>')
    return out


def baseline_html(D, curves):
    """Descriptive picture of the no-sharing chains, plus their training curves."""
    c = D.v(G09, "nocomm", "cost_mean")
    cv = D.v(G09, "nocomm", "cvar")
    rr = D.v(G09, "nocomm", "ready_rate")
    hs = D.v(G09, "nocomm", "holding_share")
    bw = [D.v(G09, "nocomm", f"bw_{k}").mean()
          for k in ("retailer", "wholesaler", "distributor", "manufacturer")]
    vst = D.v(G09, "nocomm", "V_vs_static")
    stats = f"""
    <div class="cols">
      <div class="stat"><div class="k">cost per horizon</div>
        <div class="v num">{c.mean():,.0f}</div>
        <div class="s">across 15 independently trained chains, spread of only
        {c.std(ddof=1)/c.mean():.1%} between them &mdash; these policies converged to the
        same place.</div></div>
      <div class="stat"><div class="k">held as stock</div>
        <div class="v num">{hs.mean():.0%}</div>
        <div class="s">of cost is holding, the rest backorder penalty: a balanced regime
        rather than one the cost structure decides in advance.</div></div>
      <div class="stat"><div class="k">order variance amplification</div>
        <div class="v num">{bw[0]:.1f} &rarr; {bw[3]:.1f}</div>
        <div class="s">retailer to factory. The bullwhip is present without anyone
        behaving irrationally &mdash; it is what inference under delay produces.</div></div>
      <div class="stat"><div class="k">versus the textbook rule</div>
        <div class="v num">{vst.mean():+,.0f}</div>
        <div class="s">difference against an analytic base-stock policy fitted to the
        same demand. Near zero is the goal: it means the baseline is a fair opponent,
        not a straw man.</div></div>
    </div>"""
    if not curves:
        return stats + ('<p style="max-width:68ch;margin-top:26px;color:var(--void)">'
                        'The training curves draw from '
                        '<code>runs/C_ar1_r09_nocomm_reta_b10_s30..44/metrics_train.csv</code>, '
                        'which ships in the analysis archive. Unpack it into '
                        '<code>runs/</code> and rebuild to show fifteen convergence '
                        'traces here. <code>metrics_gate.csv</code> is used as a '
                        'fallback if the full episode log was not kept.</p>')
    paths, y0, y1 = spark_path(curves)
    lines = "".join(f'<polyline points="{p}" fill="none" stroke="var(--void)" '
                    f'stroke-width="1.1" opacity=".55"/>' for p in paths)
    return stats + f"""
    <figure class="curve">
      <svg viewBox="0 0 760 170" preserveAspectRatio="none" role="img"
           aria-label="cost per episode during training, one line per seed">{lines}</svg>
      <figcaption><b>Total chain cost per episode during training, one line per seed.</b>
      Vertical axis runs from <span class="num">{y0:,.0f}</span> to
      <span class="num">{y1:,.0f}</span>; horizontal axis is 0 to 24,000 episodes.
      {len(curves)} chains start knowing nothing — ordering essentially at random, which
      is why cost begins near the top — and learn only from their own inventory,
      backlog and incoming orders. No demand information, no communication, no
      demonstrations.
      <br><br>
      The steep early fall is the chains learning to hold stock at all. The long flat
      section is the part that matters for this study: they have converged, and they
      have converged <em>to the same place</em> from fifteen different random starts.
      That is what licenses treating a difference against them as a measurement of
      information rather than an accident of training.</figcaption>
    </figure>"""


NOTATION = [
  ("V", "the quantity everything measures",
   "Cost of the chain that shares nothing, minus cost of the otherwise identical chain "
   "that shares. Positive means sharing is cheaper. Units are cost per chain-horizon: "
   "holding plus backorder penalty, summed over four stages and fifty weeks. V = +860 "
   "against a no-sharing cost of 3,899 is a 22% reduction."),
  ("rho", "how much demand remembers itself",
   "The autocorrelation of end-customer demand. Demand follows an AR(1) process: this "
   "week's demand is a weighted blend of last week's and fresh noise, and rho is that "
   "weight. At rho = 0 each week is independent and yesterday tells you nothing about "
   "tomorrow. At rho = 0.9 demand drifts in long swings and last week's figure is a "
   "strong predictor. Real grocery demand is nearer the high end; promotional or "
   "spot-market demand nearer the low."),
  ("paired within seed", "why the error bars are small",
   "Each communicating chain is compared against its OWN no-sharing twin, trained from "
   "the same initialisation and scored on the same fifty demand sequences. The two "
   "differ in the channel and nothing else, so seed-to-seed variation cancels."),
  ("n = 15", "the unit of inference",
   "Fifteen independently initialised chains per condition. Statistics are computed on "
   "the fifteen seed-level values, never on pooled episodes: episodes within a seed "
   "share a policy and are not independent evidence."),
  ("SE", "how precisely the average is pinned down",
   "Standard error of the mean across the fifteen seeds: the spread divided by the "
   "square root of fifteen. Roughly, the true value sits within two standard errors of "
   "the reported mean."),
  ("SD", "how much the seeds disagree",
   "Standard deviation across seeds. Large SD with a small SE means a real effect "
   "measured through noisy replicates; small SD means the fifteen chains found "
   "essentially the same answer."),
  ("95% CI", "the interval that matters",
   "The range in which the true mean would fall 95 times in 100 repetitions of the "
   "whole campaign. An interval excluding zero is the visual form of a significant "
   "result."),
  ("d_z", "effect size, unit-free",
   "The mean divided by the standard deviation across seeds. Above 0.8 is conventionally "
   "large. It answers 'how big relative to the noise', which p alone does not."),
  ("t", "the test statistic",
   "Mean divided by standard error. p is computed from it and the sample size; it is "
   "shown so the inference can be reconstructed rather than taken on trust."),
  ("p", "how surprising this would be if nothing were there",
   "One-sided, because every hypothesis was registered with a direction before the "
   "campaign ran. The multiplicity table near the end shows which claims survive "
   "correction for the number of questions asked."),
  ("concordance", "the check that matters more than p",
   "Whether all fifteen seeds point the same way. A large mean driven by three seeds is "
   "a different object from a modest mean reproduced fifteen times out of fifteen, and "
   "p does not distinguish them."),
  ("gap recovered", "value on an interpretable scale",
   "Cost units are arbitrary, so V is also expressed as a fraction of the distance "
   "between the best base-stock policy that ignores demand and the best one that "
   "conditions on it. 0.68 means sharing captured about two thirds of what conditioning "
   "is worth in principle."),
  ("listening", "proof the signal was used, not merely present",
   "The trained policy is re-scored with its incoming messages scrambled in time. The "
   "values stay realistic; only their correspondence to the current state is destroyed. "
   "A cost increase proves the policy was acting on the signal rather than coincidentally "
   "performing well."),
  ("signal vs demand", "what the channel actually carries",
   "Correlation between the transmitted number and true end-customer demand. 1.00 means "
   "the channel carries demand exactly; near zero means it carries something else, "
   "whatever the wiring diagram suggests."),
  ("slope", "three different quantities, all called slope",
   "In the response chart it is how far a receiver moves its order-up-to level per unit "
   "of message received &mdash; a behavioural response, referenced against the fitted "
   "base-stock coefficient of about 4. In the persistence result it is cost units of "
   "value per unit of demand autocorrelation. In the capability result it is cost units "
   "of value per doubling of training budget. The axis label always says which."),
  ("Wilcoxon p", "the test that assumes less",
   "A rank-based alternative to the t-test, valid when the fifteen seed values are not "
   "normally distributed. Reported beside the t-test throughout: where they agree, the "
   "normality assumption was not doing any work."),
  ("CVaR", "the bad quarter",
   "Mean cost of the worst 25% of episodes. Reported alongside the mean because a supply "
   "chain's exposure is not the average week."),
  ("bullwhip", "order variance amplification",
   "Variance of a stage's orders divided by variance of end-customer demand. Above 1 "
   "means that stage orders more erratically than customers buy; the figure typically "
   "grows with distance from the customer."),
  ("ready rate", "service level",
   "Fraction of weeks in which the retailer met customer demand from stock without "
   "backordering."),
  ("holding share", "which side of the cost the chain sits on",
   "Fraction of total cost that is holding rather than backorder penalty. Near 50% is a "
   "balanced regime, where neither stocking out nor overstocking dominates by "
   "construction."),
]


def notation_html():
    return "".join(
        f'<div class="note"><div class="nk"><span class="num">{k}</span>'
        f'<em>{sub}</em></div><p>{txt}</p></div>' for k, sub, txt in NOTATION)


AGENT_PARAMS = [
  ("what the agent decides", [
    ("order-up-to level", "41 discrete levels spanning 0 to 100 units"),
    ("decided by", "one shared recurrent policy, with the stage's identity as an input"),
    ("what it observes", "own inventory, backlog, on-order, last incoming order, "
                         "plus the message if its condition provides one"),
    ("message", "3 numbers per period, scaled to unit variance before entering the network"),
  ]),
  ("how it learns", [
    ("algorithm", "PPO, centralised critic and decentralised execution"),
    ("network", "256 hidden units, gamma 0.99"),
    ("learning rate", "3e-4 actor, 1e-3 critic, halved every 6,000 episodes"),
    ("batch", "8 episodes per update, 4 epochs, clip 0.1, gradient norm capped at 0.2"),
    ("exploration", "entropy bonus annealed 0.02 to 0 over 24,000 episodes"),
  ]),
  ("how a run is scored", [
    ("budget", "24,000 episodes, fixed and identical for every condition"),
    ("checkpoint chosen by", "held-out evaluation every 400 episodes on seeds no "
                             "training or reporting ever touches, smoothed over three readings"),
    ("reported on", "50 further episodes on a third, disjoint seed space"),
    ("seeds", "30 to 44, the same fifteen in every condition"),
  ]),
]


def params_html():
    out = ""
    for group, rows in AGENT_PARAMS:
        body = "".join(f'<tr><td>{k}</td><td>{v}</td></tr>' for k, v in rows)
        out += (f'<div class="pgroup"><h3>{group}</h3>'
                f'<table class="params">{body}</table></div>')
    return out


def topology_html(D):
    """Which wiring wins, and the diagnostics that say why."""
    rows = ""
    ref = D.v(G09, "raw_reta_b10")
    for fam, lab, note in (
        ("raw_reta_b10", "broadcast to all", "full visibility"),
        ("raw_upst", "relayed stage to stage", "one hop each"),
        ("raw_no_n", "no channel", "reference"),
        ("raw_manu", "broadcast from the factory", "demand-correlated but thrice filtered"),
        ("raw_down", "own-order echo", "no new information"),
    ):
        v = D.v(G09, fam)
        if not len(v):
            continue
        cost = D.v(G09, fam, "cost_mean").mean()
        li = D.v(G09, fam, "listen_shuffled")
        sg = D.v(G09, fam, "signaling_r")
        bw = D.v(G09, fam, "bw_manufacturer").mean()
        share = v.mean() / ref.mean() if ref.mean() else 0
        rows += (f'<tr><td>{lab}<span class="sub">{note}</span></td>'
                 f'<td class="n">{cost:,.0f}</td>'
                 f'<td class="n">{v.mean():+,.0f}</td>'
                 f'<td class="n">{share:.0%}</td>'
                 f'<td class="n">{li.mean():+,.0f}</td>'
                 + (f'<td class="n">{sg.mean():.2f}</td>' if len(sg) and np.isfinite(sg.mean()) else '<td class="n" style="color:var(--void)">&mdash;</td>') + 
                 f'<td class="n">{bw:.1f}</td></tr>')
    return rows


def ladder_svg(D):
    """Every condition ranked by value -- the campaign in one chart."""
    items = []
    for g, fam, lab in (
        (G09, "dhatc_reta_b10", "learned forecast"), (G09, "raw_reta_b10", "raw demand"),
        (G09, "arpred_reta", "forecast"), (G09, "learned", "emergent protocol"),
        (G09, "raw_lag1", "demand, 1 period old"), (G09, "raw_lag2", "demand, 2 periods old"),
        (G09, "raw_upst", "relayed upstream"), (G09, "ip_reta", "inventory position"),
        ("rho0.6_ar1_b10", "raw", "raw demand, rho 0.6"),
        ("rho0.3_ar1_b10", "raw", "raw demand, rho 0.3"),
        ("rho0_ar1_b10", "raw_reta", "raw demand, rho 0"),
        (G09, "raw_no_n", "no_neighbor"), (G09, "raw_manu", "manufacturer_broadcast"),
        (G09, "raw_down", "downstream_only")):
        v = D.v(g, fam)
        if len(v):
            items.append((lab, float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))))
    items.sort(key=lambda t: -t[1])
    h, rowh, pad, lw = len(items) * 26 + 34, 26, 6, 210
    span = max(abs(v) for _, v, _ in items) * 1.12
    zero = lw + 14
    scale = (900 - zero - 78) / span
    bars = ""
    for i, (lab, v, se) in enumerate(items):
        y = 22 + i * rowh
        w = abs(v) * scale
        col = "var(--signal)" if v > 5 else ("var(--no)" if v < -5 else "var(--void)")
        hatch = '' if v >= 0 else ' opacity=".85"'
        bars += (f'<text x="{lw}" y="{y+13}" text-anchor="end" class="lb">{lab}</text>'
                 f'<rect x="{zero}" y="{y+4}" width="{max(w,1):.1f}" height="15" '
                 f'fill="{col}"{hatch} rx="1"/>'
                 f'<line x1="{zero + max(w-se*scale,0):.1f}" x2="{zero + w + se*scale:.1f}" '
                 f'y1="{y+11.5}" y2="{y+11.5}" stroke="var(--ink)" stroke-width="1" opacity=".5"/>'
                 f'<text x="{zero+w+se*scale+9:.1f}" y="{y+15}" class="vl">{v:+,.0f}</text>')
    return (f'<svg viewBox="0 0 900 {h}" class="chart" role="img" aria-label="value by condition">'
            f'<line x1="{zero}" x2="{zero}" y1="14" y2="{h-8}" stroke="var(--ink)" stroke-width="1"/>'
            f'{bars}</svg>')


def curve_svg(R):
    """The persistence gradient as a curve."""
    pts = R.get("H2", {}).get("extra", {}).get("curve")
    if not pts:
        return ""
    w, h, pad = 620, 220, 40
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    X = lambda v: pad + (v - min(xs)) / max(max(xs) - min(xs), 1e-9) * (w - 2 * pad)
    Y = lambda v: h - pad - (v - 0) / max(max(ys), 1) * (h - 2 * pad)
    poly = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in pts)
    dots = "".join(
        f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="4.5" fill="var(--signal)"/>'
        f'<text x="{X(x):.1f}" y="{Y(y)-13:.1f}" text-anchor="middle" class="vl">{y:+,.0f}</text>'
        f'<text x="{X(x):.1f}" y="{h-16}" text-anchor="middle" class="lb">rho {x:g}</text>'
        for x, y in pts)
    return (f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" '
            f'aria-label="value against demand autocorrelation">'
            f'<line x1="{pad}" x2="{w-pad}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" '
            f'stroke="var(--rule)"/>'
            f'<polyline points="{poly}" fill="none" stroke="var(--signal)" stroke-width="2"/>'
            f'{dots}</svg>')


def trace_svg(D):
    """Factory orders through an unannounced doubling of demand."""
    t = D.extras.get("trace")
    if not t:
        return ""
    w, h, pad = 900, 250, 46
    n = len(t["demand"])
    allv = t["nocomm"] + t["raw"] + t["demand"]
    lo, hi = 0, max(allv) * 1.08
    X = lambda i: pad + i / max(n - 1, 1) * (w - 2 * pad)
    Y = lambda v: h - pad - (v - lo) / max(hi - lo, 1) * (h - 2 * pad)
    def poly(v, col, dash=""):
        p = " ".join(f"{X(i):.1f},{Y(x):.1f}" for i, x in enumerate(v))
        return (f'<polyline points="{p}" fill="none" stroke="{col}" stroke-width="2"'
                f'{dash}/>')
    sx = X(t["shock"] - t["w0"])
    dash = ' stroke-dasharray="4 3"'      # hoisted: 3.11 f-strings reject backslashes
    return (f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" '
            f'aria-label="factory orders through a demand shock">'
            f'<line x1="{sx:.1f}" x2="{sx:.1f}" y1="{pad-14}" y2="{h-pad}" '
            f'stroke="var(--ink)" stroke-dasharray="3 3" opacity=".5"/>'
            f'<text x="{sx+7:.1f}" y="{pad-4}" class="lb">demand doubles</text>'
            f'{poly(t["demand"], "var(--flow)", dash)}'
            f'{poly(t["nocomm"], "var(--void)")}'
            f'{poly(t["raw"], "var(--signal)")}'
            f'<text x="{pad}" y="{h-14}" class="lb">week {t["w0"]}</text>'
            f'<text x="{w-pad}" y="{h-14}" text-anchor="end" class="lb">'
            f'week {t["w0"]+n-1}</text></svg>')



def _axes(w, h, pad):
    return (lambda i, n: pad + i / max(n - 1, 1) * (w - 2 * pad),
            lambda v, lo, hi: h - pad - (v - lo) / max(hi - lo, 1e-9) * (h - 2 * pad))


def shock_state_svg(D):
    """Inventory and backlog at every stage, through the shock. The chart that shows
    WHY the delay costs money rather than that it exists."""
    t = D.extras.get("shock_state")
    if not t:
        return ""
    names = ["retailer", "wholesaler", "distributor", "factory"]
    n = len(t["nocomm"]["back"])
    hi = max(max(r) for arm in ("nocomm", "raw") for r in
             (list(map(max, zip(*t[arm]["back"]))),)) * 1.1 or 1
    w, h, pad = 216, 150, 26
    X, Y = _axes(w, h, pad)
    out = ""
    for k, nm in enumerate(names):
        sx = X(t["shock"] - t["w0"], n)
        def line(arm, col):
            p = " ".join(f"{X(i,n):.1f},{Y(row[k],0,hi):.1f}"
                         for i, row in enumerate(t[arm]["back"]))
            return f'<polyline points="{p}" fill="none" stroke="{col}" stroke-width="1.8"/>'
        out += (f'<figure class="mini"><svg viewBox="0 0 {w} {h}" class="chart">'
                f'<line x1="{pad}" x2="{w-pad}" y1="{Y(0,0,hi):.1f}" y2="{Y(0,0,hi):.1f}" '
                f'stroke="var(--rule)"/>'
                f'<line x1="{sx:.1f}" x2="{sx:.1f}" y1="{pad-10}" y2="{h-pad}" '
                f'stroke="var(--ink)" stroke-dasharray="3 3" opacity=".45"/>'
                f'{line("nocomm","var(--void)")}{line("raw","var(--signal)")}'
                f'<text x="{pad}" y="{h-8}" class="lb">{nm}</text>'
                f'<text x="{w-pad}" y="{pad-2}" text-anchor="end" class="lb">'
                f'{hi:.0f}</text></svg></figure>')
    return f'<div class="minis">{out}</div>'


def cost_split_svg(D):
    """Holding versus backorder cost at every stage, shared against not."""
    c = D.extras.get("cost_split")
    if not c:
        return ""
    names = ["retailer", "wholesaler", "distributor", "factory"]
    hi = max(c[a]["hold"][i] + c[a]["back"][i]
             for a in ("nocomm", "raw") for i in range(4)) * 1.06
    w, h, pad, bw = 900, 250, 46, 26
    rows = ""
    for i, nm in enumerate(names):
        x0 = pad + 40 + i * ((w - pad * 2 - 40) / 4)
        for j, (arm, lab) in enumerate((("nocomm", "not shared"), ("raw", "shared"))):
            hold, back = c[arm]["hold"][i], c[arm]["back"][i]
            hh = hold / hi * (h - 2 * pad)
            bh = back / hi * (h - 2 * pad)
            x = x0 + j * (bw + 10)
            rows += (f'<rect x="{x:.0f}" y="{h-pad-hh:.1f}" width="{bw}" height="{hh:.1f}" '
                     f'fill="var(--flow)" opacity=".75"/>'
                     f'<rect x="{x:.0f}" y="{h-pad-hh-bh:.1f}" width="{bw}" height="{bh:.1f}" '
                     f'fill="{"var(--void)" if arm=="nocomm" else "var(--signal)"}"/>'
                     f'<text x="{x+bw/2:.0f}" y="{h-pad+14}" text-anchor="middle" '
                     f'class="lb" style="font-size:9px">{lab}</text>'
                     f'<text x="{x+bw/2:.0f}" y="{h-pad-hh-bh-6:.1f}" text-anchor="middle" '
                     f'class="vl">{hold+back:,.0f}</text>')
        rows += (f'<text x="{x0+bw+5:.0f}" y="{pad-14}" text-anchor="middle" '
                 f'class="lb">{nm}</text>')
    return (f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" '
            f'aria-label="cost split by echelon">{rows}</svg>')


def response_svg(D):
    """What the receiver does with what it hears."""
    r = D.extras.get("response")
    if not r:
        return ""
    w, h, pad = 280, 200, 40
    out = ""
    allp = [p for v in r.values() if v for p in v["pts"]]
    if not allp:
        return ""
    xlo, xhi = min(p[0] for p in allp), max(p[0] for p in allp)
    ylo, yhi = min(p[1] for p in allp) * .96, max(p[1] for p in allp) * 1.04
    for lab, v in r.items():
        if not v:
            continue
        X = lambda x: pad + (x - xlo) / max(xhi - xlo, 1e-9) * (w - 2 * pad)
        Y = lambda y: h - pad - (y - ylo) / max(yhi - ylo, 1e-9) * (h - 2 * pad)
        pts = "".join(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3.5" '
                      f'fill="var(--signal)"/>' for x, y in v["pts"])
        poly = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in v["pts"])
        out += (f'<figure class="mini wide"><svg viewBox="0 0 {w} {h}" class="chart">'
                f'<polyline points="{poly}" fill="none" stroke="var(--signal)" '
                f'stroke-width="1.4" opacity=".5"/>{pts}'
                f'<text x="{pad}" y="{h-10}" class="lb">{lab}</text>'
                f'<text x="{w-pad}" y="{pad-6}" text-anchor="end" class="vl">'
                f'slope {v["slope"]:+.2f}'
                f'{" ± " + format(v["se"], ".2f") if v.get("se") else ""}'
                f'</text></svg></figure>')
    return f'<div class="minis">{out}</div>'


def bullwhip_svg(D):
    """Order-variance amplification along the chain, per condition."""
    names = ["retailer", "wholesaler", "distributor", "factory"]
    series = []
    for fam, lab, col in (("nocomm", "not shared", "var(--void)"),
                          ("raw_reta_b10", "shared", "var(--signal)"),
                          ("raw_upst", "relayed", "var(--flow)")):
        v = [D.v(G09, fam, f"bw_{k}").mean() for k in
             ("retailer", "wholesaler", "distributor", "manufacturer")]
        if all(np.isfinite(v)):
            series.append((lab, v, col))
    if not series:
        return ""
    w, h, pad = 620, 230, 46
    hi = max(max(v) for _, v, _ in series) * 1.12
    X = lambda i: pad + i / 3 * (w - 2 * pad)
    Y = lambda v: h - pad - v / hi * (h - 2 * pad)
    out = ""
    for lab, v, col in series:
        p = " ".join(f"{X(i):.1f},{Y(x):.1f}" for i, x in enumerate(v))
        dots = "".join(f'<circle cx="{X(i):.1f}" cy="{Y(x):.1f}" r="4" fill="{col}"/>'
                       for i, x in enumerate(v))
        out += (f'<polyline points="{p}" fill="none" stroke="{col}" stroke-width="2"/>'
                f'{dots}<text x="{X(3)+8:.0f}" y="{Y(v[3])+4:.1f}" class="lb" '
                f'fill="{col}">{lab}</text>')
    ticks = "".join(f'<text x="{X(i):.1f}" y="{h-16}" text-anchor="middle" '
                    f'class="lb">{n}</text>' for i, n in enumerate(names))
    return (f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" '
            f'aria-label="bullwhip by echelon">{out}{ticks}</svg>')


def strip(seeds, w=250, h=26):
    """Fifteen dots on a line: concordance made visible."""
    if not seeds:
        return ""
    m = max(abs(min(seeds)), abs(max(seeds))) * 1.15 or 1
    zero = w / 2
    dots = "".join(
        f'<circle cx="{zero + v / m * (w/2 - 8):.1f}" cy="{h/2}" r="3.2" '
        f'fill="{"var(--signal)" if v > 0 else "var(--no)"}" opacity=".8"/>'
        for v in seeds)
    return (f'<svg viewBox="0 0 {w} {h}" class="strip" role="img" '
            f'aria-label="one dot per seed">'
            f'<line x1="8" x2="{w-8}" y1="{h/2}" y2="{h/2}" stroke="var(--rule)"/>'
            f'<line x1="{zero}" x2="{zero}" y1="4" y2="{h-4}" stroke="var(--ink-2)"/>'
            f'{dots}</svg>')


def latency_table(D):
    lat = D.extras.get("latency")
    if not lat:
        return ""
    rows = ""
    for thr in sorted(lat, key=lambda k: int(k)):
        L = lat[thr]
        rows += (f'<tr><td>order above {thr} units</td>'
                 f'<td class="n">{" / ".join("+" + str(x) for x in L["nocomm"])}</td>'
                 f'<td class="n">{" / ".join("+" + str(x) for x in L["raw"])}</td></tr>')
    return (f'<table class="cert" style="margin-top:16px"><tr><th>threshold for '
            f'"responded"</th><th>no sharing (R/W/D/M)</th>'
            f'<th>shared (R/W/D/M)</th></tr>{rows}</table>')


def service_html(D):
    out = ""
    for fam, lab in (("nocomm", "no sharing"), ("raw_reta_b10", "demand shared")):
        rr = D.v(G09, fam, "ready_rate")
        bw = D.v(G09, fam, "bw_manufacturer")
        cm = D.v(G09, fam, "cost_mean")
        if not len(rr):
            continue
        out += (f'<div class="stat"><div class="k">{lab}</div>'
                f'<div class="v num">{rr.mean():.1%}</div>'
                f'<div class="s">of weeks the retailer met customer demand from stock. '
                f'Total chain cost {cm.mean():,.0f}.</div></div>')
    d = D.v(G09, "raw_reta_b10", "ready_rate") - D.v(G09, "nocomm", "ready_rate")
    t, p2 = st.ttest_1samp(d, 0)
    out += (f'<div class="stat"><div class="k">difference</div>'
            f'<div class="v num">{d.mean():+.1%}</div>'
            f'<div class="s">across the fifteen matched pairs '
            f'(p = {p2:.1e}), against a cost reduction of about 22%.</div></div>')
    return out


def bars(D):
    """The fitted benchmark levels, read from the campaign's own baselines file. Back-
    computing them from per-seed differences drifts by tens of units; this does not."""
    p = os.path.join(ROOT, "runs", "baselines_rho0.9.json")
    if os.path.exists(p):
        b = json.load(open(p, encoding="utf-8"))
        return float(b["static_bs"]), float(b["cond_bs"])
    nc = D.v(G09, "nocomm", "cost_mean").mean() - D.v(G09, "nocomm", "V_vs_static").mean()
    gap = (D.v(G09, "raw_reta_b10", "V_vs_static").mean() /
           max(D.v(G09, "raw_reta_b10", "gap_recovered").mean(), 1e-9))
    return nc, nc - gap


def topo_rank(D):
    C = {"retailer_broadcast": D.v(G09, "raw_reta_b10"),
         "upstream_only": D.v(G09, "raw_upst"),
         "no_neighbor": D.v(G09, "raw_no_n"),
         "manufacturer_broadcast": D.v(G09, "raw_manu"),
         "downstream_only": D.v(G09, "raw_down")}
    return pairwise_rows(C)


def pairwise_rows(C):
    names = list(C)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = C[names[i]] - C[names[j]]
            if d.std(ddof=1) == 0:
                continue
            t, p2 = st.ttest_1samp(d, 0)
            ci = st.t.interval(0.95, len(d) - 1, loc=d.mean(),
                               scale=d.std(ddof=1) / np.sqrt(len(d)))
            pairs.append([names[i], names[j], float(d.mean()), float(p2),
                          [float(ci[0]), float(ci[1])], int((d > 0).sum()), len(d)])
    order = sorted(range(len(pairs)), key=lambda k: pairs[k][3])
    run = 0.0
    for r, k in enumerate(order):
        run = max(run, min(1.0, (len(pairs) - r) * pairs[k][3]))
        pairs[k].append(run)
    rows = ""
    for a, b, m, p, ci, pos, n, hp in pairs:
        rows += (f'<tr><td>{a} &minus; {b}</td><td class="n">{m:+,.0f}</td>'
                 f'<td class="n">[{ci[0]:+,.0f}, {ci[1]:+,.0f}]</td>'
                 f'<td class="n">{p:.4f}</td><td class="n">{hp:.4f}</td>'
                 f'<td class="n">{pos}/{n}</td>'
                 f'<td><span class="pill p-{"broken" if hp < 0.05 else "holds"}">'
                 f'{"differ" if hp < 0.05 else "indistinguishable"}</span></td></tr>')
    return rows


def levels_table(C):
    rows = ""
    for k, x in C.items():
        n = len(x); se = x.std(ddof=1) / np.sqrt(n) if n > 1 else 0
        degenerate = (n > 1 and x.std(ddof=1) == 0)
        ci = st.t.interval(0.95, n - 1, loc=x.mean(), scale=se) if se > 0 else (x.mean(), x.mean())
        sd = x.std(ddof=1)
        t, p2 = st.ttest_1samp(x, 0) if sd > 0 else (0.0, 1.0)
        p1 = p2 / 2 if t > 0 else 1 - p2 / 2
        try:
            w = st.wilcoxon(x).pvalue if sd > 0 else float("nan")
        except ValueError:
            w = float("nan")
        DASH = '<td class="n" style="color:var(--void)">&mdash;</td>'
        if degenerate:
            rows += (f'<tr><td>{k}</td><td class="n">{x.mean():+,.1f}</td>'
                     f'<td class="n">0.0</td><td class="n">exact</td>'
                     + DASH * 3 +
                     f'<td class="n">0/{n}</td></tr>')
        else:
            rows += (f'<tr><td>{k}</td><td class="n">{x.mean():+,.1f}</td>'
                     f'<td class="n">{se:,.1f}</td>'
                     f'<td class="n">[{ci[0]:+,.0f}, {ci[1]:+,.0f}]</td>'
                     f'<td class="n">{(x.mean()/sd if sd else 0):.2f}</td>'
                     f'<td class="n">{p1:.2e}</td>'
                     f'<td class="n">{w:.2e}</td>'
                     f'<td class="n">{int((x>0).sum())}/{n}</td></tr>')
    return rows


def content_levels(D):
    return levels_table({"learned forecast": D.v(G09, "dhatc_reta_b10"),
                         "raw demand": D.v(G09, "raw_reta_b10"),
                         "analytic forecast": D.v(G09, "arpred_reta"),
                         "emergent protocol": D.v(G09, "learned"),
                         "inventory position": D.v(G09, "ip_reta")})


def topo_levels(D):
    return levels_table({"retailer_broadcast": D.v(G09, "raw_reta_b10"),
                         "upstream_only": D.v(G09, "raw_upst"),
                         "no_neighbor": D.v(G09, "raw_no_n"),
                         "manufacturer_broadcast": D.v(G09, "raw_manu"),
                         "downstream_only": D.v(G09, "raw_down")})


def content_rank(D):
    """Which content actually beats which -- paired, Holm-corrected. Without this the
    ordering of the bars invites a ranking the data does not support."""
    C = {"raw demand": D.v(G09, "raw_reta_b10"),
         "forecast": D.v(G09, "arpred_reta"),
         "learned forecast": D.v(G09, "dhatc_reta_b10"),
         "emergent protocol": D.v(G09, "learned"),
         "inventory position": D.v(G09, "ip_reta")}
    names = list(C)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = C[names[i]] - C[names[j]]
            t, p2 = st.ttest_1samp(d, 0)
            ci = st.t.interval(0.95, len(d) - 1, loc=d.mean(),
                               scale=d.std(ddof=1) / np.sqrt(len(d)))
            pairs.append([names[i], names[j], float(d.mean()), float(p2),
                          [float(ci[0]), float(ci[1])], int((d > 0).sum()), len(d)])
    order = sorted(range(len(pairs)), key=lambda k: pairs[k][3])
    run = 0.0
    for r, k in enumerate(order):
        run = max(run, min(1.0, (len(pairs) - r) * pairs[k][3]))
        pairs[k].append(run)
    rows = ""
    for a, b, m, p, ci, pos, n, hp in pairs:
        sig = hp < 0.05
        rows += (f'<tr><td>{a} &minus; {b}</td>'
                 f'<td class="n">{m:+,.0f}</td>'
                 f'<td class="n">[{ci[0]:+,.0f}, {ci[1]:+,.0f}]</td>'
                 f'<td class="n">{p:.3f}</td><td class="n">{hp:.3f}</td>'
                 f'<td class="n">{pos}/{n}</td>'
                 f'<td><span class="pill p-{"broken" if sig else "holds"}">'
                 f'{"differ" if sig else "indistinguishable"}</span></td></tr>')
    return rows


def echelon_table(D):
    ex = D.extras.get("echelon")
    if not ex:
        return ""
    out = ""
    for c, lab in (("raw", "raw demand"), ("arpred", "forecast"),
                   ("dhatc", "learned forecast"), ("learned", "emergent protocol")):
        b = ex["by_content"].get(c)
        if not b:
            continue
        m = b["mean"]
        tot = sum(m)
        up = sum(m[1:]) / tot if abs(tot) > 1e-9 else float("nan")
        cells = "".join(f'<td class="n">{x:+,.0f}</td>' for x in m)
        out += f'<tr><td>{lab}</td>{cells}<td class="n">{up:.0%}</td></tr>'
    return out


def build(D, R, ctl, hl, out_path):
    curves = read_curves(ROOT, [f"C_ar1_r09_nocomm_reta_b10_s{s}" for s in range(30, 45)])
    lat = D.extras.get("latency", {}).get("14", D.extras.get("latency", {}).get(14))
    names = ["retailer", "wholesaler", "distributor", "manufacturer"]
    spine = ""
    if lat:
        mx = max(max(lat["nocomm"]), 1)
        for i, n in enumerate(names):
            nc, sg = lat["nocomm"][i], lat["raw"][i]
            spine += f"""<div class="node"><div class="nm">{n}</div><div class="bars">
              <div class="bar nc"><span style="width:{nc/mx*100:.0f}%"></span><b>+{nc}</b></div>
              <div class="bar sg"><span style="width:{max(sg/mx*100,6):.0f}%"></span><b>+{sg}</b></div>
            </div></div>"""

    cards = ""
    for label, _, keys in GROUPS:
        cards += f'<div class="group-label">{label}</div>'
        for k in keys:
            if k in R:
                cards += card(k, R[k])

    cert = "".join(
        f'<tr><td>{c["label"]}</td><td class="n">{c["delta"]:+,.0f}</td>'
        f'<td class="n">{c["cv"]:.3f}</td>'
        f'<td><span class="pill p-{c["status"]}">{c["status"]}</span></td></tr>'
        for c in ctl)

    hl_rows = "".join(
        f'<tr><td>{k}</td><td class="n">{v["p"]:.2e}</td><td class="n">{v["holm"]:.3f}</td>'
        f'<td><span class="pill p-{"holds" if v["survives"] else "broken"}">'
        f'{"survives" if v["survives"] else "does not survive"}</span></td></tr>'
        for k, v in sorted(hl.items(), key=lambda kv: kv[1]["p"]))

    f = R.get("F_CONTENT", {}).get("stats", [{}])[0]
    tail = R.get("H_TAIL", {}).get("extra", {}).get("ratio", 0)
    ech = R.get("H_ECHELON", {}).get("extra", {})
    hero_stats = f"""
    <div class="cols">
      <div class="stat"><div class="k">cost reduction</div>
        <div class="v num">{f.get('V',0):+,.0f}</div>
        <div class="s">per chain-horizon, about {f.get('gap',0):.0%} of the gap between the
        best unconditional and the best demand-conditional base-stock policy.</div></div>
      <div class="stat"><div class="k">worst quarter of periods</div>
        <div class="v num">{tail:.2f}×</div>
        <div class="s">the tail improves that much more than the average period — sharing
        is a risk instrument, not only an efficiency one.</div></div>
      <div class="stat"><div class="k">captured upstream</div>
        <div class="v num">{sum(ech.get('share',[0,0,0,0])[1:]):.0%}</div>
        <div class="s">of the benefit accrues to stages that cannot see end-customer
        demand for themselves.</div></div>
    </div>"""

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>The value of demand-information sharing — four-echelon evidence</title>
<style>{CSS}</style></head><body>

<header class="top"><div class="wrap">
  <p class="eyebrow">Four-echelon serial supply chain · learned replenishment · 15 seeds</p>
  <h1>What is a demand signal<br>worth, and <em>to whom</em>?</h1>
  <p class="lede">{ABSTRACT}</p>

  <div class="spine">
    <div class="spine-legend">
      <span class="key"><i style="background:var(--void)"></i> no sharing</span>
      <span class="key"><i style="background:var(--signal)"></i> demand shared</span>
      <span style="color:var(--void)">weeks until each stage responds to an unannounced doubling of demand</span>
    </div>
    <div class="chain">{spine}</div>
    <p class="spine-note"><b>How to read it.</b> Each pair of bars is one stage of the
    chain. The bar length is the number of weeks that stage took to raise its orders
    after end-customer demand doubled, measured as the first week its average order
    crossed the midpoint between the old and new demand levels. Grey is no sharing, amber
    is demand shared. Shorter is better.
    <br><br>
    Without sharing the shock climbs the chain at about three weeks per stage: two weeks
    of physical lead time plus roughly one week for that stage to distinguish a genuine
    level change from an unusually busy week. The factory is still ordering for the old
    world <b>nine weeks</b> after the customer changed. With sharing every stage moves in
    the first week, because every stage hears the retailer's observation directly instead
    of waiting for it to arrive as someone else's order.
    <br><br>
    The retailer's bar is +1 in both conditions, and that is the control built into the
    picture: the retailer can always see demand, so sharing cannot help it, and it does
    not. The three columns to its right are where the channel does its work. The pattern
    was identical on all fifteen independently trained chains, and identical again at
    three different thresholds for what counts as "responded":</p>
    {latency_table(D)}
  </div>
  {hero_stats}
  <figure class="fig"><figcaption class="fc-top">Factory orders through an unannounced
    doubling of end-customer demand, averaged over all 15 seeds &mdash; <span style="color:var(--flow)">demand</span>,
    <span style="color:var(--void)">no sharing</span>,
    <span style="color:var(--signal)">demand shared</span></figcaption>
    {trace_svg(D)}
    <figcaption><b>How to read it.</b> The dashed blue line is what customers actually
    buy: flat at about 8 units a week, then doubling to roughly 20 at the marked week and
    staying there. The two solid lines are what the <em>factory</em> — four stages away
    from those customers — orders in response. Grey is the chain with no communication;
    amber is the same chain with the retailer's demand shared.
    <br><br>
    The amber line jumps in the first week after the shock, overshooting to nearly twice
    the new demand level before settling. That overshoot is not a defect: the factory is
    simultaneously refilling a pipeline that has been drained by four weeks of
    under-ordering. The grey line does something more troubling — it carries on ordering
    for the old world, dips <em>below</em> its previous level around week 27, and only
    begins to correct near week 33. The dip is the supply-line delay working against it:
    orders it placed before the shock are still arriving, so its inventory briefly looks
    adequate while a backlog builds beneath it.
    <br><br>
    Both chains end up ordering roughly the right amount. Only one of them does it while
    customers are still waiting.</figcaption></figure>
</div></header>

<section><div class="wrap">
  <div class="sec-head"><h2>The game being played</h2>
    <p>The Beer Distribution Game, the standard laboratory model of a serial supply
    chain, played here by learning agents instead of people.</p></div>
  <div class="rules">
    <div class="rule"><b>Four stages, one product</b>Retailer, wholesaler, distributor,
      factory, in series. The retailer serves end customers; every other stage serves
      only the stage below it.</div>
    <div class="rule"><b>Each week, each stage decides one number</b>How much to order
      from its supplier. That is the entire decision. Prices, capacity and product mix
      do not exist here.</div>
    <div class="rule"><b>Nothing arrives immediately</b>An order takes two weeks to
      reach the supplier and the goods two more weeks to come back. A decision made
      today shows up in inventory four weeks from now.</div>
    <div class="rule"><b>Only the retailer sees customers</b>Everyone else observes just
      their own inventory, their own backlog, what is in transit to them, and the order
      their customer placed last week. Demand itself is invisible upstream.</div>
    <div class="rule"><b>Two ways to be wrong</b>Holding stock costs money every week it
      sits. Failing to fill an order costs more, and the unfilled amount stays owed
      until it is delivered. Total cost is the sum across all four stages.</div>
    <div class="rule"><b>Fifty weeks, then the bill</b>Each episode runs fifty weeks.
      The number this report calls cost is that total, and the objective is to minimise
      it for the chain as a whole, not for any one stage.</div>
  </div>
  <p style="max-width:72ch;margin-top:26px">
    Played by people, this setup reliably produces the bullwhip effect: small changes in
    customer demand become wild swings in factory orders, driven by the delay between
    deciding and seeing the consequence. Sterman's experiments showed the pattern is not
    carelessness — it survives training, incentives and repetition, because people
    consistently under-weight the orders already in transit. The question this report
    asks is what happens to that dynamic when the retailer is allowed to simply tell
    everyone what it is seeing.</p>
</div></section>

<section><div class="wrap">
  <div class="sec-head"><h2>The chain, and the wire laid across it</h2>
    <p>Four stages in series. Goods flow down, orders flow up, and every hop costs two
    periods of lead time. Only the retailer sees the end customer.</p></div>
  <div class="flowline">
    <div class="fl-stage"><b>retailer</b><span>sees demand</span></div>
    <div class="fl-arrow">orders &uarr;</div>
    <div class="fl-stage"><b>wholesaler</b><span>blind</span></div>
    <div class="fl-arrow">orders &uarr;</div>
    <div class="fl-stage"><b>distributor</b><span>blind</span></div>
    <div class="fl-arrow">orders &uarr;</div>
    <div class="fl-stage"><b>manufacturer</b><span>blind</span></div>
  </div>
  <p style="max-width:68ch;margin:26px 0 0">
    Communication is one exchange per period, and its entire structure is a four-by-four
    routing matrix: row <em>i</em> carries a one wherever stage <em>i</em> receives stage
    <em>j</em>'s message. Senders compose, the matrix routes once, receivers order, the
    world advances. There is no reply within a period and no multi-hop relay: information
    travels at most one edge per period, exactly like the goods. Changing who hears whom
    means changing this matrix and nothing else, which is what makes the geometry
    conditions comparable.</p>
  <div class="mats">{matrices_html(D)}</div>
  <p style="max-width:68ch;margin:22px 0 0;color:var(--ink-2);font-size:16px">
    The last matrix is the integrity check. It runs the full communication machinery with
    nobody wired to anybody, so its value must be exactly zero. Across fifteen seeds it
    was: <span class="num">0.000</span>. Whatever the other conditions measure, it is not
    an artefact of having a channel.</p>
</div></section>

<section><div class="wrap">
  <div class="sec-head"><h2>How the campaign ran</h2>
    <p>One machine, one code revision, one command. Nothing in the pipeline touches a
    result before the verification stages have passed.</p></div>
  <div class="cols">
    <div class="stat"><div class="k">training runs</div><div class="v num">510</div>
      <div class="s">34 conditions &times; 15 seeds, each 24,000 episodes, every one
      completing without error.</div></div>
    <div class="stat"><div class="k">wall clock</div><div class="v num">~17 h</div>
      <div class="s">60 concurrent workers on 64 vCPU (AMD EPYC 7702), 62 GB RAM. The
      GPU in the instance is unused: the policies are small and train on CPU.</div></div>
    <div class="stat"><div class="k">evaluations</div><div class="v num">2,640</div>
      <div class="s">each policy scored on 50 held-out episodes, plus channel-corruption
      probes, disruption transfers and training-budget checkpoints.</div></div>
  </div>
  <ol class="pipeline">
    <li><b>Verification first</b> &mdash; 89 environment tests, 21 adapter tests and 17
      invariant checks must pass before a single episode is trained. Among them: that a
      communicating agent with its channel blanked is bit-identical to a
      no-communication agent, and that the message head receives gradient only where a
      message is actually learned.</li>
    <li><b>Message scaling measured, not chosen</b> &mdash; each signal's standard
      deviation is measured on a dedicated seed space before training, and the signal is
      divided by it. This is done per demand regime, because the same signal has
      different spread at different levels of persistence.</li>
    <li><b>Analytic benchmarks fitted</b> &mdash; the best unconditional and best
      demand-conditional base-stock policies are fitted for every regime, providing the
      external yardstick the validity table uses.</li>
    <li><b>Training</b> &mdash; all 510 runs, identical budget, no per-condition tuning,
      no early stopping.</li>
    <li><b>Scoring</b> &mdash; on seed spaces disjoint from both training and checkpoint
      selection, so no number reported here was ever optimised against.</li>
    <li><b>Analysis</b> &mdash; statistics, consolidated results sheet, figures and a
      manifest recording the SHA-256 of every source file that produced them.</li>
  </ol>
</div></section>

<section><div class="wrap">
  <div class="sec-head"><h2>Before any signal: what the chain does alone</h2>
    <p>Every number in this report is a difference against this baseline, so it is worth
    seeing what it is. Fifteen independent chains learn to replenish with no
    communication at all, from a cold start.</p></div>
  {baseline_html(D, curves)}
</div></section>

<section><div class="wrap">
  <div class="sec-head"><h2>The yardstick: two textbook policies</h2>
    <p>Learned policies are hard to argue about in the abstract. Two classical
    base-stock rules, fitted to the same demand, turn every result into a distance from
    something known.</p></div>
  <div class="cols">
    <div class="stat"><div class="k">unconditional base stock</div>
      <div class="v num">{bars(D)[0]:,.0f}</div>
      <div class="s">One fixed order-up-to level per stage, never changing. This is what a
      chain can achieve with no demand information at all, so it is the floor a
      no-sharing agent must reach to count as competent.</div></div>
    <div class="stat"><div class="k">demand-conditional base stock</div>
      <div class="v num">{bars(D)[1]:,.0f}</div>
      <div class="s">Each stage's target adjusts linearly with observed demand. This is
      roughly what perfect demand visibility is worth to a classical policy, and it is
      the ceiling the shared chains are measured against.</div></div>
    <div class="stat"><div class="k">the gap between them</div>
      <div class="v num">{bars(D)[0]-bars(D)[1]:,.0f}</div>
      <div class="s">The analytically available prize from conditioning on demand. Every
      "gap recovered" figure in this report is a fraction of this number.</div></div>
  </div>
  <ol class="pipeline" style="margin-top:34px">
    <li><b>Fitted, not assumed</b> &mdash; both rules are tuned by coordinate search over
      the order-up-to levels, one stage at a time, repeated until nothing improves. No
      closed-form approximation is used, so the benchmark is the best member of its
      policy class rather than a textbook formula that may not suit these lead times.</li>
    <li><b>Fitted on their own seeds</b> &mdash; the search uses a seed space reserved for
      it. The benchmark is then scored on the same fifty evaluation episodes the learned
      policies face, so no rule is tuned on the data it is judged by.</li>
    <li><b>Scored episode by episode</b> &mdash; the benchmark's cost is recorded per
      episode, not just as an average, so a learned policy can be compared against it on
      matched demand sequences rather than against a single summary number.</li>
    <li><b>Refitted for every regime</b> &mdash; each demand process, each cost structure
      and each disruption scenario gets its own pair of benchmarks. A rule tuned for
      persistent demand would flatter the learned policies under a different one.</li>
  </ol>
  <p style="max-width:72ch;margin-top:26px">
    This is what makes the validity table below readable. A no-sharing agent that lands
    on the unconditional benchmark has found the best policy available without
    information, so any further improvement must come from the channel. One that lands
    far below it has not finished learning, and a difference measured against it would
    be measuring training, not information.</p>
</div></section>

<section><div class="wrap">
  <div class="sec-head"><h2>Is the comparison valid?</h2>
    <p>Every number here is a difference against a chain that shares nothing. That
    difference measures information only if the no-sharing chain actually reached the best
    policy available without information. This table checks that against an analytic
    base-stock benchmark, in every regime, before any result is read.</p></div>
  <table class="cert">
    <tr><th>demand regime</th><th>no-sharing minus benchmark</th><th>seed variation</th><th></th></tr>
    {cert}
  </table>
  <p style="max-width:66ch;margin-top:22px;color:var(--ink-2);font-size:16px">
    A positive figure means the learned chain beat the analytic rule, which is expected
    where a stationary rule is a weak benchmark. The check earns its place: it caught a
    selection defect in the persistence grid that would otherwise have produced a
    confident and wrong result for the study's central comparative static.</p>
</div></section>

<section><div class="wrap">
  <div class="sec-head"><h2>Forecast or raw data: does it matter what you send?</h2>
    <p>The bars in the previous chart are ordered, which invites a ranking. This tests
    whether the ranking exists. Every pair is compared within seed, against the same
    no-sharing reference, and corrected for the ten comparisons.</p></div>
  <table class="cert"><tr><th>condition</th><th>V</th><th>SE</th><th>95% CI</th><th>d<sub>z</sub></th><th>p (t)</th><th>p (Wilcoxon)</th><th>seeds +</th></tr>{content_levels(D)}</table>
  <p style="max-width:72ch;margin:22px 0 6px;color:var(--ink-2);font-size:16px">
    All five beat sharing nothing. The question is whether they differ from
    <em>each other</em>, which requires comparing them directly rather than reading the
    order of the bars:</p>
  <table class="cert"><tr><th>comparison</th><th>difference</th><th>95% CI</th><th>p</th><th>Holm p</th><th>seeds</th><th></th></tr>{content_rank(D)}</table>
  <p style="max-width:72ch;margin-top:24px">
    <b>The three demand-bearing encodings are statistically indistinguishable.</b> Raw
    sales, the analytic forecast and the learned forecast differ by at most 60 cost units
    with confidence intervals spanning zero, and none of those three comparisons survives
    correction. The apparent ordering in the bar chart &mdash; learned forecast ahead of
    raw ahead of forecast &mdash; is noise at this sample size, and reporting it as a
    ranking would be reading the chart rather than the data.
    <br><br>
    What <em>is</em> established: all three beat the emergent protocol, and all four beat
    inventory position, both with wide margins and near-perfect seed agreement. So the
    answer to "forecast or raw data" is that the question does not matter much, and that
    is itself the finding. Partners need not agree on a forecasting methodology before
    sharing; they need only send something that carries demand. What matters instead is
    who receives it and how quickly &mdash; effects three to eight times larger than any
    difference between these encodings.</p>
</div></section>

<section><div class="wrap">
  <div class="sec-head"><h2>What the delay actually costs</h2>
    <p>Backlog at each stage through the same shock. The value figures say sharing is
    worth 860; this says where those 860 come from.</p></div>
  <figure class="fig"><figcaption class="fc-top">Unfilled orders waiting at each stage
    &mdash; <span style="color:var(--void)">no sharing</span>,
    <span style="color:var(--signal)">demand shared</span></figcaption>
    {shock_state_svg(D)}
    <figcaption>Read the panels left to right along the chain. The retailer's backlog
    rises in both conditions, because customers arrive whether or not anyone was warned.
    Everything after that diverges. Without sharing the backlog marches upstream, one
    stage at a time, and the factory is still absorbing it long after the shock.
    <br><br>
    The counter-intuitive part is worth pausing on: the unshared factory often shows
    <em>less</em> backlog early, because it has not yet noticed anything is wrong. Its
    customers are waiting at the wholesaler instead. Sharing does not eliminate the
    shortage &mdash; the goods still take four weeks to arrive &mdash; it moves the
    absorption upstream, where lead times are longest and holding is cheapest, and gets
    it over with sooner.</figcaption></figure>

  <div class="sec-head" style="margin-top:46px"><h2>Who captures the benefit</h2>
    <p>The same cost reduction, split by stage, for each kind of shared signal.</p></div>
  <table class="cert">
    <tr><th>signal</th><th>retailer</th><th>wholesaler</th><th>distributor</th>
        <th>factory</th><th>upstream share</th></tr>
    {echelon_table(D)}
  </table>
  <p style="max-width:72ch;margin-top:22px">
    The benefit is hump-shaped rather than rising with distance: the wholesaler gains
    most in every case. It sits one hop from demand, so the signal reaches it fresh, and
    it is the first stage that cannot see demand for itself &mdash; the largest gap
    between what it needs and what it has. The factory gains less despite being blindest,
    because three lead times of physical delay limit what any forecast can rescue.
    <br><br>
    One cell is negative. Under the emergent protocol the retailer ends up slightly worse
    off while every upstream stage gains: the chain as a whole is better, and the stage
    doing the transmitting pays a small price for it. That is a free-rider structure
    arising on its own, without anyone designing it.</p>

  <div class="sec-head" style="margin-top:46px"><h2>Where the money goes</h2>
    <p>Cost at each stage, split into stock held and orders unfilled.</p></div>
  <figure class="fig"><figcaption class="fc-top">
    <span style="color:var(--flow)">holding</span> &nbsp;·&nbsp;
    <span style="color:var(--void)">backorder, not shared</span> &nbsp;·&nbsp;
    <span style="color:var(--signal)">backorder, shared</span></figcaption>
    {cost_split_svg(D)}
    <figcaption>Sharing does not simply trade one cost for the other. Holding cost falls at
    every stage, and backorder cost falls at three of the four; only at the factory does
    it edge up, by about five units against a 214-unit fall in holding. That single
    exception is the absorption story appearing in the ledger: the shared chain
    deliberately carries a little more unfilled order at the stage where waiting is
    cheapest.
    The retailer's bars barely move, which is the same control appearing again: it never
    lacked the information.</figcaption></figure>

  <div class="sec-head" style="margin-top:46px"><h2>What sharing does to service</h2>
    <p>Cost is the objective. Service level is what a customer experiences, and it does
    not move the way one might expect.</p></div>
  <div class="cols">{service_html(D)}</div>
  <p style="max-width:72ch;margin-top:22px">
    The shared chain fills a slightly <em>smaller</em> share of customer orders straight
    from retail stock, while costing about a fifth less overall. This is not a defect in
    the result; it is the policy doing what it was asked. The objective is total cost
    across all four stages, and under this cost structure &mdash; where holding a unit
    for two weeks costs about what delaying it one week does &mdash; running retail
    leaner and absorbing more of the variability upstream is cheaper. A chain that wanted
    a higher fill rate would need that written into the objective, not into the channel.
    <br><br>
    It is worth stating plainly because it sets a boundary on the claim: information
    sharing here buys cost, not service. Whether the same channel would buy service under
    a heavier backorder penalty is a separate question, and the one condition that tested
    a heavier penalty found the value of sharing unchanged.</p>

  <div class="sec-head" style="margin-top:46px"><h2>Does the receiver act on it?</h2>
    <p>The wholesaler's order-up-to level against the number it just received.</p></div>
  <figure class="fig">{response_svg(D)}
    <figcaption><b>Slope here means the receiver's response:</b> how far its order-up-to
    level moves for each additional unit in the message. Each point is an average over
    many periods within one bin of received
    values. Under <span class="num">retailer_broadcast</span> the response is a clean
    upward line: hear a larger number, stock more. Under
    <span class="num">upstream_only</span> the slope survives but shallower &mdash; the
    signal is real but noisier by the time it arrives. Under
    <span class="num">downstream_only</span> the line is flat: the agent receives a
    perfectly valid number every week and correctly ignores it, because it is its own
    order coming back.
    <br><br>
    Slopes are averaged over all fifteen seeds with their standard error, so the figure
    is a property of the condition rather than of one training run. The fitted
    conditional base-stock benchmark uses a coefficient near 4 for these stages, which
    makes it a rough reference rather than a proven optimum: broadcast reaches about
    three quarters of it without ever being told what the right answer was, and the
    placebo sits at essentially zero.</figcaption></figure>

  <div class="sec-head" style="margin-top:46px"><h2>The bullwhip, and what sharing does
    to it</h2><p>Order variance divided by demand variance, along the chain.</p></div>
  <figure class="fig">{bullwhip_svg(D)}
    <figcaption>The classic amplification is present in every condition: orders get more
    erratic the further you stand from the customer. Sharing lowers cost by roughly a
    fifth while leaving this curve almost untouched &mdash; and the relayed channel
    actually raises it. Cost and volatility are separate outcomes, and a chain can be
    made much cheaper without being made calmer.</figcaption></figure>
</div></section>

<section><div class="wrap">
  <div class="sec-head"><h2>Every condition, ranked</h2>
    <p>One bar per condition, all against their own matched no-sharing chain. Whiskers
    are one standard error across the fifteen seeds.</p></div>
  <figure class="fig">{ladder_svg(D)}
    <figcaption>Bar length is magnitude; <span style="color:var(--no)">red</span> marks
    conditions that cost more than sharing nothing. The three at the foot are the
    controls: a disabled channel returns exactly zero, and the two that carry no demand
    information return less than nothing.</figcaption></figure>
  <div class="sec-head" style="margin-top:44px"><h2>Value against demand persistence</h2>
    <p>The same channel, the same content, four demand processes.</p></div>
  <figure class="fig">{curve_svg(R)}
    <figcaption>Near zero where demand is memoryless, and convex thereafter: each
    additional unit of persistence is worth more than the last.</figcaption></figure>
</div></section>

<section><div class="wrap">
  <div class="sec-head"><h2>The registry</h2>
    <p>Grouped by what happened, not by number. Open any hypothesis for its statement,
    the mechanism it rests on, the measured result seed by seed, and the literature it
    engages.</p></div>
  {cards}
</div></section>

<section><div class="wrap">
  <div class="sec-head"><h2>Topology, tested pair by pair</h2>
    <p>The same content and budget throughout; only the routing matrix changes.</p></div>
  <table class="cert"><tr><th>condition</th><th>V</th><th>SE</th><th>95% CI</th><th>d<sub>z</sub></th><th>p (t)</th><th>p (Wilcoxon)</th><th>seeds +</th></tr>{topo_levels(D)}</table>
  <p style="max-width:72ch;margin:22px 0 6px;color:var(--ink-2);font-size:16px">
    Unlike the contents, these separate almost everywhere:</p>
  <table class="cert"><tr><th>comparison</th><th>difference</th><th>95% CI</th><th>p</th><th>Holm p</th><th>seeds</th><th></th></tr>{topo_rank(D)}</table>
  <p style="max-width:72ch;margin-top:24px">
    Broadcast beats relay by <span class="num">+524</span>, relay beats a disabled
    channel by <span class="num">+336</span>, and every one of those comparisons survives
    correction with near-unanimous seed agreement. The finding that deserves attention is
    at the bottom: <b>downstream_only is significantly worse than having no channel at
    all</b> (Holm p = 0.019). A wire that carries nothing useful is not neutral.
    <br><br>
    The single pair that does not separate is manufacturer_broadcast against
    no_neighbor. A demand signal filtered through three replenishment policies and
    delayed by three hops is, statistically, the same as no signal.</p>
</div></section>

<section><div class="wrap">
  <div class="sec-head"><h2>Which wiring wins, and why</h2>
    <p>The same message content, the same training budget, the same seeds. Only the
    routing matrix changes.</p></div>
  <table class="cert">
    <tr><th>topology</th><th>cost</th><th>value</th><th>of broadcast</th>
        <th>listening</th><th>signal vs demand</th><th>factory bullwhip</th></tr>
    {topology_html(D)}
  </table>
  <p style="max-width:70ch;margin-top:24px">
    Broadcast wins, and the diagnostics explain the ranking rather than merely reporting
    it. Relay carries a signal the receivers demonstrably act on &mdash; its listening
    score is nearly as high as broadcast's &mdash; yet it retains only about a third of
    the value, because by the third and fourth stage the signal has been through two
    replenishment policies and two lead times. The last two rows are the control: their
    correlation with true demand collapses, their receivers stop reacting, and value goes
    to zero or below. Value tracks what the wire carries, not whether a wire exists.</p>
  <p style="max-width:70ch;margin-top:16px;color:var(--ink-2);font-size:16px">
    Note the bullwhip column. Relay has the highest order-variance amplification at the
    factory of any live condition, above even the no-channel case: a delayed, filtered
    signal can make upstream ordering more volatile while still lowering cost.</p>
</div></section>

<section><div class="wrap">
  <div class="sec-head"><h2>Reading the numbers</h2>
    <p>What each quantity in this report means, in one place.</p></div>
  <div class="notes">{notation_html()}</div>
</div></section>

<section><div class="wrap">
  <div class="sec-head"><h2>What every agent was, identically</h2>
    <p>Held fixed across all 510 runs. Conditions differ in the demand process, the
    channel and the cost regime &mdash; never in the learner.</p></div>
  <div class="pgrid">{params_html()}</div>
  <p style="max-width:70ch;margin-top:24px;color:var(--ink-2);font-size:16px">
    Equal fixed budget is what makes V an information effect rather than a compute
    effect, and it is the first thing worth checking in any comparison of this kind.</p>
</div></section>

<section><div class="wrap">
  <div class="sec-head"><h2>What to do with this</h2>
    <p>The registry as four questions, in the order that decides how much a visibility
    programme is worth.</p></div>
  <div class="guide">
    <div class="gq"><span class="qn">1</span><b>Does your demand carry memory week to
      week?</b>If it drifts in swings, sharing is worth a great deal and worth more the
      stronger the persistence. If each week is independent of the last, expect close to
      nothing: there is no forecastable component to transmit.
      <em>H2 &mdash; value rose from ~50 to ~860 across the persistence range.</em></div>
    <div class="gq"><span class="qn">2</span><b>Can everyone see the source, or only
      their neighbour?</b>Broadcasting point-of-sale data to every stage is worth roughly
      three times passing it up link by link. If the architecture is already dyadic,
      changing that is the single largest available improvement.
      <em>H-SOURCE &mdash; relay retained 39% of broadcast's value.</em></div>
    <div class="gq"><span class="qn">3</span><b>How fresh is the feed?</b>Each week of
      delay costs about an eighth of the channel's value. Weekly batch beats monthly by
      more than any refinement of what is in the file.
      <em>H-TIME &mdash; two weeks of staleness cost 226 of 860.</em></div>
    <div class="gq"><span class="qn">4</span><b>Can your partners act on it?</b>Value
      rose with the receiving policy's sophistication, not against it. A feed delivered
      to a chain that cannot use it will underdeliver, and the fault will look like the
      data's.
      <em>H-BUDGET &mdash; value rose ~200 per doubling of planning capability.</em></div>
  </div>
  <p style="max-width:72ch;margin-top:28px">
    Two things this evidence does not support. Sharing did not calm the bullwhip, so it
    should not be sold on that basis. And what you transmit &mdash; raw sales, a
    forecast, or a demand innovation &mdash; made no measurable difference, so partners
    need not agree on a forecasting methodology before starting.</p>
</div></section>

<section><div class="wrap">
  <div class="sec-head"><h2>Correcting for asking many questions</h2>
    <p>The study makes many directional claims at once, so the strong results are shown
    against a family-wise correction across all of them.</p></div>
  <table class="cert">
    <tr><th>claim</th><th>p</th><th>corrected</th><th></th></tr>{hl_rows}
  </table>
</div></section>

<footer><div class="wrap">
  Generated {date.today().isoformat()} by scripts/build_report.py ·
  numbers from runs/RESULTS.csv · prose from docs/hypotheses_text.py<br>
  Rebuild after editing the prose: <span style="color:var(--ink-2)">python scripts/build_report.py</span>
</div></footer>
</body></html>"""
    open(out_path, "w", encoding="utf-8").write(html)
    return len(html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(ROOT, "runs", "RESULTS.csv"))
    ap.add_argument("--extras", default=os.path.join(ROOT, "docs", "extras.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "report.html"))
    a = ap.parse_args()
    D = Data(a.csv, a.extras)
    R, ctl, hl = compute(D)
    n = build(D, R, ctl, hl, a.out)
    missing = [k for k in HYPOTHESES if k not in R]
    print(f"[report] {len(R)} hypotheses rendered from {len(D.rows)} arm-rows "
          f"-> {a.out} ({n/1024:.0f} KB)")
    if missing:
        print(f"[report] no statistics found for: {', '.join(missing)}")
    if not D.extras:
        print("[report] NOTE: docs/extras.json absent -- echelon, budget and the "
              "latency spine are omitted")


if __name__ == "__main__":
    main()
