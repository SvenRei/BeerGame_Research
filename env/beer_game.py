"""env/beer_game.py -- ADAPTER ONLY. Zero physics live in this file.

The environment is the user's validated implementation, vendored UNMODIFIED at:
    vendor/envs/beer_game_env.py        (BeerGameParallelEnv)
    vendor/scripts/demand_families.py   (AR(1) / NegBin family subclasses)
    vendor/conf/config.yaml             (reference config, for provenance)

This module translates that PettingZoo ParallelEnv into the small numpy interface
signal_lab consumes. Every state variable, cost, lead time, pipeline and demand draw is
produced by the vendored code. If this adapter and the vendored env ever disagree, the
vendored env is correct by definition.

Translation performed here (interface only, never physics):
  * actions: signal_lab emits INTEGER order quantities [N]; the vendored env expects a
    dict of float actions in [0,1] scaled by max_order. We divide by max_order; the
    env's own round-half-up recovers the integer exactly (asserted in tests/test_env.py).
  * returns: the PettingZoo 5-tuple becomes (obs [N,4], local_costs [N], done, info).
    Costs are read from the env's own infos[agent]["local_cost"] -- never recomputed.
  * demand: info["demand"] is the retailer's realized incoming order for the period,
    read from the env's current_incoming_order.
  * AR(1) is selected the way the vendored code requires -- demand_type="poisson" plus
    family="ar1" via make_demand_family_envs -- NOT demand_type="ar1", which the
    vendored env rejects by design.

The vendored env primes its pipelines at reset (4 units in each of two shipment slots
and two order slots; manufacturer one order slot) giving initial on-order 16/16/16/12.
That is the env's own behaviour, inherited here rather than reimplemented.
"""
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from vendor.envs.beer_game_env import MAX_DELAY, BeerGameParallelEnv  # noqa: E402
from vendor.scripts.demand_families import ar1_step  # noqa: E402,F401  (re-exported)
from vendor.scripts.demand_families import make_demand_family_envs  # noqa: E402

AGENTS = ["retailer", "wholesaler", "distributor", "manufacturer"]
N_AGENTS = 4
OBS_DIM = 4                       # [inventory, backlog, on_order, last_incoming]
PIPE_SLOTS = MAX_DELAY            # the vendored get_global_state walks MAX_DELAY slots
STATE_DIM = 1 + N_AGENTS * (3 + 2 * MAX_DELAY)   # matches get_global_state exactly

_AR1Env, _NegBinEnv, _FamilyEnv = make_demand_family_envs(BeerGameParallelEnv)

# Adapter defaults. Keys are passed through to the vendored env, so its own defaults
# and validation apply to anything omitted here.
DEFAULTS = dict(
    horizon=50, max_order=100, holding_cost=0.5, backorder_cost=1.0,
    demand_family="ar1", ar1_mu=12.0, ar1_rho=0.9, ar1_sigma=3.0,
    dr_lambda_lo=4.0, dr_lambda_hi=24.0,   # dr_poisson: per-episode lambda ~ U[lo,hi]
    poisson_mu=8.0, jittery_lead_time=False,
)

_ADAPTER_ONLY_KEYS = ("demand_family", "ar1_rho", "ar1_mu", "ar1_sigma",
                      "poisson_mu", "rho")


class BeerGame:
    """Numpy-facing adapter over the vendored BeerGameParallelEnv."""

    def __init__(self, config=None):
        self.cfg = {**DEFAULTS, **(config or {})}
        family = self.cfg.get("demand_family", "ar1")
        # black_swan / extreme_chaos are the VENDORED stress decks. They are fixed
        # calendars (the clock explains ~75% of demand variance, residual autocorr ~0),
        # so TRAINING on them is confounded: an agent with memory learns the schedule
        # and no message can add anything. They are exposed here for ZERO-SHOT OOD
        # EVALUATION of policies trained on ar1/poisson, where the schedule is genuinely
        # unanticipated and the retailer's observation IS an early warning.
        if family not in ("ar1", "poisson", "dr_poisson",
                          "black_swan", "extreme_chaos"):
            raise ValueError(f"demand_family must be one of ar1 | poisson | dr_poisson "
                             f"| black_swan | extreme_chaos, got {family!r}")

        env_cfg = {k: v for k, v in self.cfg.items() if k not in _ADAPTER_ONLY_KEYS}
        env_cfg["demand_type"] = "poisson"      # all families ride on this; see docstring
        if family in ("black_swan", "extreme_chaos"):
            env_cfg["demand_type"] = family     # the vendored deck, unmodified
            self._env = BeerGameParallelEnv(env_cfg)
        elif family == "dr_poisson":
            # P1's regime-uncertainty side: the VENDORED FamilyRandomizedBeerGame draws
            # a fresh lambda ~ U[dr_lambda_lo, dr_lambda_hi] at every reset. Restricted
            # to the poisson family so the treatment is exactly "unknown rate".
            env_cfg.update(dr_families=["poisson"],
                           dr_lambda_lo=float(self.cfg["dr_lambda_lo"]),
                           dr_lambda_hi=float(self.cfg["dr_lambda_hi"]))
            self._env = _FamilyEnv(env_cfg)
        elif family == "ar1":
            env_cfg.update(family="ar1",
                           ar1_mu=float(self.cfg["ar1_mu"]),
                           ar1_rho=float(self.cfg["ar1_rho"]),
                           ar1_sigma=float(self.cfg["ar1_sigma"]))
            self._env = _AR1Env(env_cfg)
        else:
            self._env = BeerGameParallelEnv(env_cfg)
            mu = float(self.cfg["poisson_mu"])
            if mu != 8.0:
                # The vendored base env hardcodes poisson(8). Honour poisson_mu by
                # overriding ONLY the demand draw; all physics stay untouched.
                env_ref = self._env

                def _roll(step, _e=env_ref, _mu=mu):
                    return float(_e.np_random.poisson(_mu))
                self._env._roll_stochastic_demand = _roll

        self.h = float(self.cfg["holding_cost"])
        self.b = float(self.cfg["backorder_cost"])
        self.horizon = int(self.cfg["horizon"])
        self.max_order = int(self.cfg["max_order"])
        self.last_demand = 0.0

    # ------------------------------------------------------------------ views
    @property
    def inventory(self):
        return self._env.inventory

    @property
    def backlog(self):
        return self._env.backlog

    @property
    def on_order(self):
        return self._env.unfulfilled_orders

    @property
    def last_incoming(self):
        return self._env.current_incoming_order

    @property
    def t(self):
        """Steps completed so far (0 after reset)."""
        return int(self._env.current_step)

    def _obs(self):
        return np.stack([self._env._build_obs(a) for a in AGENTS]).astype(np.float32)

    def global_state(self):
        return self._env.get_global_state()

    # ------------------------------------------------------------------ lifecycle
    def reset(self, seed=None):
        self._env.reset(seed=seed)
        self.last_demand = 0.0
        return self._obs()

    def step(self, orders):
        """orders: array-like [N] of integer order quantities in [0, max_order].
        Returns (obs [N,4], local_costs [N], done, info)."""
        o = np.clip(np.asarray(orders, dtype=float), 0, self.max_order)
        actions = {a: np.array([float(o[i]) / self.max_order], dtype=np.float32)
                   for i, a in enumerate(AGENTS)}
        _obs, _rew, _term, truncs, infos = self._env.step(actions)
        costs = np.array([float(infos[a]["local_cost"]) for a in AGENTS],
                         dtype=np.float32)
        self.last_demand = float(self._env.current_incoming_order["retailer"])
        done = bool(any(truncs.values()))
        return self._obs(), costs, done, {"demand": self.last_demand, "t": self.t}

    @staticmethod
    def inventory_position(obs_row):
        """IP = inventory - backlog + on_order, from one raw observation row."""
        return float(obs_row[0]) - float(obs_row[1]) + float(obs_row[2])
