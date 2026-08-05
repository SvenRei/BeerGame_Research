"""signal_lab/curves.py -- fig14 (training curve) + fig15 (six-panel diagnostics).

fig15 panels: gate cost + monitor(0.9) + bests | policy entropy | action std (grid
spread) | HONEST explained variance (the canary line at ev_canary) | value loss |
approx KL. The EV panel is first-class because an invisible dead critic cost the
legacy project seven tuning rounds.
"""
import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) if r[k] not in ("", "nan") else np.nan
                         for r in rows]) for k in rows[0]} if rows else {}


def fig14(tag, out_dir):
    tr = _read(os.path.join(ROOT, "runs", tag, "metrics_train.csv"))
    if not tr:
        return None
    ep, c = tr["episode"], tr["team_cost"]
    w = min(50, max(1, len(c) // 10))
    roll = np.convolve(c, np.ones(w) / w, mode="valid")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ep[w - 1:], roll, lw=1.4, label=tag.lstrip("_") or tag)  # mpl drops labels starting with "_"
    i = int(np.argmin(roll))
    ax.plot(ep[w - 1:][i], roll[i], "o", ms=6)
    ax.set_xlabel("Training episode")
    ax.set_ylabel(f"Team cost per episode ({w}-ep rolling mean, training env)")
    ax.set_title(f"TRAINING curve -- {tag} (stochastic policy; dot = minimum)")
    ax.legend(); ax.grid(alpha=0.3)
    p = os.path.join(out_dir, f"fig14_train_{tag}.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    return p


def fig15(tag, out_dir, ev_canary=0.05):
    g = _read(os.path.join(ROOT, "runs", tag, "metrics_gate.csv"))
    u = _read(os.path.join(ROOT, "runs", tag, "metrics_update.csv"))
    if not g or not u:
        return None
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"training diagnostics -- {tag}")
    a = ax[0, 0]
    a.plot(g["episode"], g["gate_cost"], lw=1.2, label="gate (.15/.45/.75)")
    a.plot(g["episode"], g["monitor_rho09"], lw=1.0, alpha=0.7, label="monitor rho=0.9")
    nb = g["gate_cost"] <= g["best"] + 1e-9
    a.plot(g["episode"][nb], g["gate_cost"][nb], "r.", ms=6)
    a.set_title("gate + monitor + new bests"); a.legend(fontsize=8)
    ax[0, 1].plot(u["episode"], u["entropy"], "g-", lw=0.9)
    ax[0, 1].set_title("policy entropy")
    ax[0, 2].plot(u["episode"], u["state_absmean"], lw=0.8, alpha=0.6,
                  label="|state| mean")
    ax2 = ax[0, 2].twinx()
    ax2.plot(u["episode"], _read_col(u, "grad_norm"), color="C1", lw=0.6, alpha=0.7)
    ax[0, 2].set_title("|state| (left) / grad norm (right)"); ax[0, 2].legend(fontsize=8)
    a = ax[1, 0]
    a.plot(u["episode"], u["honest_ev"], lw=1.0, color="C3")
    a.axhline(ev_canary, ls="--", color="gray", lw=1)
    a.set_ylim(-0.5, 1.0); a.set_title("HONEST explained variance (canary dashed)")
    ax[1, 1].semilogy(u["episode"], np.maximum(u["value_loss"], 1e-8), lw=0.7)
    ax[1, 1].set_title("value loss")
    ax[1, 2].plot(u["episode"], u["approx_kl"], lw=0.6, color="purple")
    ax[1, 2].set_title("approx KL")
    for row in ax:
        for a_ in row:
            a_.set_xlabel("episode"); a_.grid(alpha=0.3)
    p = os.path.join(out_dir, f"fig15_diag_{tag}.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    return p


def _read_col(d, k):
    return d.get(k, np.full_like(d["episode"], np.nan))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True, help="comma-separated run tags")
    ap.add_argument("--out", default=os.path.join(ROOT, "figs"))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    for tag in [t.strip() for t in a.arms.split(",")]:
        for fn in (fig14, fig15):
            p = fn(tag, a.out)
            print(f"[curves] {'wrote ' + p if p else 'SKIP (missing CSVs) ' + tag}")


if __name__ == "__main__":
    main()
