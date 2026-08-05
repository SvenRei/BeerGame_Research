"""scripts/fit_forecaster.py -- fit the frozen one-step demand forecaster for dhatc.

Trains a tiny GRU to predict d_t/100 from the sequence of d_{t-1}/100 on simulated
AR(1) demand (same sampler as the env), then saves assets/forecaster_ar1r9.pt.
The provider loads it FROZEN (requires_grad=False); this script is the only place its
weights ever change. Certification check printed at the end: RMSE must beat the
unconditional predictor (mu) and approach the analytic AR(1) conditional bound.

  python scripts/fit_forecaster.py --rho 0.9
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.beer_game import ar1_step  # noqa: E402
from signal_lab.messages import FrozenForecaster  # noqa: E402


def simulate(rho, mu, sigma, n_seqs, T, seed):
    rng = np.random.default_rng(seed)
    out = np.zeros((T, n_seqs), dtype=np.float32)
    for j in range(n_seqs):
        latent = mu
        for t in range(T):
            out[t, j], latent = ar1_step(latent, mu, rho, sigma, rng)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, default=0.9)
    ap.add_argument("--mu", type=float, default=12.0)
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    torch.manual_seed(0)
    model = FrozenForecaster(hidden=16)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    d = simulate(a.rho, a.mu, a.sigma, n_seqs=64, T=51, seed=7) / 100.0
    x = torch.tensor(d[:-1]).unsqueeze(-1)           # d_{t-1}  [T, B, 1]
    y = torch.tensor(d[1:]).unsqueeze(-1)            # d_t
    for k in range(int(a.steps)):
        out, _ = model.gru(x)
        pred = model.head(out)
        loss = torch.nn.functional.mse_loss(pred, y)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        dv = simulate(a.rho, a.mu, a.sigma, n_seqs=32, T=51, seed=99) / 100.0
        xv = torch.tensor(dv[:-1]).unsqueeze(-1)
        yv = torch.tensor(dv[1:]).unsqueeze(-1)
        out, _ = model.gru(xv)
        rmse = float(torch.sqrt(torch.nn.functional.mse_loss(model.head(out), yv))) * 100
        base = float(np.sqrt(np.mean((dv[1:] * 100 - a.mu) ** 2)))
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = a.out or os.path.join(root, "assets", f"forecaster_ar1r{a.rho:g}".replace("0.", "") + ".pt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "hidden": 16,
                "rho": a.rho, "mu": a.mu, "sigma": a.sigma,
                "val_rmse": rmse, "uncond_rmse": base}, out_path)
    print(f"[forecaster] val RMSE {rmse:.2f} vs unconditional {base:.2f} "
          f"(analytic innovation sd = {a.sigma:.2f}) -> {out_path}")
    if rmse >= base:
        print("[forecaster] WARNING: not better than the unconditional predictor -- "
              "do NOT certify this checkpoint")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
