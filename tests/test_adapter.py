"""tests/test_adapter.py -- does env/beer_game.py faithfully expose the vendored env?

The PHYSICS are covered by tests/test_vendor_env.py (the user's own 89-test suite, run
verbatim against vendor/envs/beer_game_env.py). This file tests only the translation
layer: action scaling, cost pass-through, observation/state pass-through, demand-family
selection, and episode semantics. Every assertion compares the adapter against the
vendored env driven directly -- the vendored env is the reference, always.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.beer_game import (AGENTS, N_AGENTS, OBS_DIM, STATE_DIM,  # noqa: E402
                           BeerGame, ar1_step)
from vendor.envs.beer_game_env import BeerGameParallelEnv  # noqa: E402
from vendor.scripts.demand_families import make_demand_family_envs  # noqa: E402

_AR1, _, _ = make_demand_family_envs(BeerGameParallelEnv)


def raw_ar1(cfg=None, seed=42):
    base = {"horizon": 50, "max_order": 100, "holding_cost": 0.5,
            "backorder_cost": 1.0, "jittery_lead_time": False,
            "demand_type": "poisson", "family": "ar1",
            "ar1_mu": 12.0, "ar1_rho": 0.9, "ar1_sigma": 3.0}
    base.update(cfg or {})
    e = _AR1(base)
    e.reset(seed=seed)
    return e


class TestTranslationEquivalence(unittest.TestCase):
    """The adapter must reproduce the vendored env step for step."""

    def test_01_full_episode_is_identical_to_the_vendored_env(self):
        A = BeerGame(); A.reset(seed=7)
        R = raw_ar1(seed=7)
        rng = np.random.default_rng(3)
        for t in range(50):
            o = rng.integers(0, 60, size=N_AGENTS)
            obsA, costA, doneA, infoA = A.step(o)
            _, _, _, truncs, infosR = R.step(
                {a: np.array([float(o[i]) / 100.0], dtype=np.float32)
                 for i, a in enumerate(R.possible_agents)})
            costR = np.array([float(infosR[a]["local_cost"]) for a in AGENTS])
            np.testing.assert_allclose(costA, costR, atol=0, rtol=0,
                                       err_msg=f"cost mismatch at t={t}")
            for a in AGENTS:
                self.assertEqual(A.inventory[a], R.inventory[a], f"inv {a} t={t}")
                self.assertEqual(A.backlog[a], R.backlog[a], f"backlog {a} t={t}")
                self.assertEqual(A.on_order[a], R.unfulfilled_orders[a], f"oo {a} t={t}")
            obsR = np.stack([R._build_obs(a) for a in AGENTS])
            np.testing.assert_array_equal(obsA, obsR)
            np.testing.assert_array_equal(A.global_state(), R.get_global_state())
            self.assertEqual(doneA, any(truncs.values()))
            self.assertEqual(infoA["demand"], float(R.current_incoming_order["retailer"]))
        self.assertTrue(doneA, "episode must be done after horizon steps")

    def test_02_integer_orders_survive_the_float_action_round_trip(self):
        """Adapter divides by max_order; the env's round-half-up must recover it."""
        for q in list(range(0, 101, 7)) + [1, 99, 100]:
            A = BeerGame(); A.reset(seed=1)
            A.step(np.full(N_AGENTS, q))
            # the new order arrives at current_step(1) + order_lead(2) = key 3;
            # keys 1 and 2 hold the env's own priming, so key 3 isolates it.
            got = A._env.order_pipelines["retailer"].pipeline.get(3, 0)
            self.assertEqual(got, q, f"order {q} became {got}")

    def test_03_orders_are_clipped_to_bounds(self):
        A = BeerGame(); A.reset(seed=1)
        A.step(np.array([500, -20, 0, 100]))
        self.assertEqual(A._env.order_pipelines["retailer"].pipeline.get(3, 0), 100)
        self.assertEqual(A._env.order_pipelines["wholesaler"].pipeline.get(3, 0), 0)
        # manufacturer order lead is 1 -> arrives at key 2
        self.assertEqual(A._env.order_pipelines["manufacturer"].pipeline.get(2, 0), 100)

    def test_04_max_order_is_respected_when_reconfigured(self):
        A = BeerGame({"max_order": 40}); A.reset(seed=1)
        A.step(np.full(N_AGENTS, 999))
        self.assertEqual(A._env.order_pipelines["retailer"].pipeline.get(3, 0), 40)


class TestContract(unittest.TestCase):
    """Shapes, dtypes and constants signal_lab relies on."""

    def test_10_observation_shape_and_dtype(self):
        A = BeerGame(); obs = A.reset(seed=42)
        self.assertEqual(obs.shape, (N_AGENTS, OBS_DIM))
        self.assertEqual(obs.dtype, np.float32)

    def test_11_global_state_matches_declared_dim(self):
        A = BeerGame(); A.reset(seed=42)
        s = A.global_state()
        self.assertEqual(s.shape, (STATE_DIM,))
        self.assertEqual(s.dtype, np.float32)

    def test_12_initial_ledger_is_the_vendored_priming(self):
        A = BeerGame(); A.reset(seed=42)
        self.assertEqual({a: A.on_order[a] for a in AGENTS},
                         {"retailer": 16, "wholesaler": 16,
                          "distributor": 16, "manufacturer": 12})
        for a in AGENTS:
            self.assertEqual(A.inventory[a], 12)
            self.assertEqual(A.backlog[a], 0)

    def test_13_inventory_position_helper(self):
        row = np.array([10.0, 3.0, 5.0, 99.0])
        self.assertEqual(BeerGame.inventory_position(row), 12.0)

    def test_14_step_counter_advances_and_horizon_truncates(self):
        A = BeerGame({"horizon": 10}); A.reset(seed=1)
        self.assertEqual(A.t, 0)
        for i in range(9):
            _, _, done, info = A.step(np.full(N_AGENTS, 4))
            self.assertFalse(done, f"done early at step {i}")
            self.assertEqual(info["t"], i + 1)
        self.assertTrue(A.step(np.full(N_AGENTS, 4))[2])

    def test_15_costs_are_finite_and_non_negative(self):
        A = BeerGame(); A.reset(seed=8)
        rng = np.random.default_rng(2)
        for _ in range(50):
            _, c, done, _ = A.step(rng.integers(0, 40, size=N_AGENTS))
            self.assertTrue(np.all(np.isfinite(c)))
            self.assertTrue(np.all(c >= 0))
            if done:
                break


class TestDemandFamilies(unittest.TestCase):
    def test_20_ar1_is_routed_through_the_family_subclass(self):
        A = BeerGame({"demand_family": "ar1", "ar1_rho": 0.9})
        self.assertEqual(A._env._config.get("family"), "ar1")
        self.assertEqual(A._env._config.get("demand_type"), "poisson")

    def test_21_ar1_mean_and_autocorrelation(self):
        A = BeerGame({"ar1_rho": 0.9}); A.reset(seed=17)
        d = []
        for _ in range(50):
            d.append(A.step(np.full(N_AGENTS, 4))[3]["demand"])
        for extra in range(7):
            A.reset(seed=18 + extra)
            for _ in range(50):
                d.append(A.step(np.full(N_AGENTS, 4))[3]["demand"])
        d = np.array(d)
        self.assertAlmostEqual(d.mean(), 12.0, delta=2.0)
        self.assertGreater(np.corrcoef(d[:-1], d[1:])[0, 1], 0.4)

    def test_22_rho_zero_is_near_white_noise(self):
        A = BeerGame({"ar1_rho": 0.0}); A.reset(seed=17)
        d = np.array([A.step(np.full(N_AGENTS, 4))[3]["demand"] for _ in range(50)])
        self.assertLess(abs(np.corrcoef(d[:-1], d[1:])[0, 1]), 0.45)

    def test_23_poisson_family_is_selectable_and_has_the_right_mean(self):
        A = BeerGame({"demand_family": "poisson", "poisson_mu": 8.0})
        self.assertIsNone(A._env._config.get("family"))
        d = []
        for s in range(8):
            A.reset(seed=100 + s)
            d += [A.step(np.full(N_AGENTS, 4))[3]["demand"] for _ in range(50)]
        self.assertAlmostEqual(float(np.mean(d)), 8.0, delta=1.0)

    def test_24_poisson_mu_override_is_honoured(self):
        A = BeerGame({"demand_family": "poisson", "poisson_mu": 20.0})
        d = []
        for s in range(8):
            A.reset(seed=200 + s)
            d += [A.step(np.full(N_AGENTS, 4))[3]["demand"] for _ in range(50)]
        self.assertAlmostEqual(float(np.mean(d)), 20.0, delta=2.0)

    def test_25_invalid_family_is_rejected(self):
        with self.assertRaises(ValueError):
            BeerGame({"demand_family": "negbin"})

    def test_26_ar1_step_is_the_vendored_sampler(self):
        from vendor.scripts.demand_families import ar1_step as vendored
        self.assertIs(ar1_step, vendored)


class TestDeterminism(unittest.TestCase):
    def test_30_same_seed_gives_identical_trajectories(self):
        runs = []
        for _ in range(2):
            A = BeerGame(); A.reset(seed=1234)
            rng = np.random.default_rng(0)
            runs.append([(float(A.step(rng.integers(0, 20, size=N_AGENTS))[1].sum()))
                         for _ in range(30)])
        self.assertEqual(runs[0], runs[1])

    def test_31_different_seeds_diverge(self):
        seqs = []
        for seed in (1, 2):
            A = BeerGame(); A.reset(seed=seed)
            seqs.append([A.step(np.full(N_AGENTS, 4))[3]["demand"] for _ in range(30)])
        self.assertNotEqual(seqs[0], seqs[1])

    def test_32_reset_does_not_pollute_global_numpy_rng(self):
        np.random.seed(7); before = np.random.random()
        np.random.seed(7); BeerGame().reset(seed=99); after = np.random.random()
        self.assertEqual(before, after)

    def test_33_observation_is_pure(self):
        A = BeerGame(); A.reset(seed=5); A.step(np.full(N_AGENTS, 4))
        a = [A._obs().tolist() for _ in range(3)]
        self.assertEqual(a[0], a[1])
        self.assertEqual(a[1], a[2])


if __name__ == "__main__":
    unittest.main(verbosity=2)
