"""env/beer_game.py -- the 4-echelon Beer Game, self-contained (numpy only).

Faithful port of the physics of the validated BeerGame_Comm environment:
  * agents: retailer -> wholesaler -> distributor -> manufacturer
  * step phases, in fixed order:
      PHASE 1  RECEIVE  goods scheduled to arrive now land in inventory
      PHASE 2  FULFILL  each stage sees incoming demand (retailer: customer AR(1);
                        others: orders arriving up the order pipeline), ships what it
                        can (remainder -> backlog) DOWN the chain; the manufacturer
                        turns received orders into production (-> own shipment pipe)
      PHASE 3  ORDER    integer order in [0, max_order] travels UP after the order lead
      PHASE 4  COST     h * inventory + b * backlog at every stage
  * lead times (defaults reproduce the original constants): order 2 (manufacturer 1),
    shipping 2, production 2
  * demand: AR(1) latent d*_t = mu + rho (d*_{t-1} - mu) + N(0, sigma); emitted
    demand = max(0, round(latent)); the latent stays unclipped so autocorrelation is
    preserved (identical to the validated ar1_step).

Deliberate simplifications vs the source (API only, physics unchanged):
  * actions are integer order quantities [N] directly (no [0,1] float indirection)
  * no PettingZoo wrapper; reset/step return plain numpy arrays
  * global-state pipeline horizon K=6 slots (max configured lead is 2; the original
    exposed 15 mostly-empty slots)

CRN contract: all randomness flows through one np.random.Generator seeded in reset();
identical seeds => bit-identical trajectories for identical action sequences.
"""
import numpy as np

AGENTS = ["retailer", "wholesaler", "distributor", "manufacturer"]
N_AGENTS = 4
OBS_DIM = 4                      # [inventory, backlog, on_order, last_incoming_order]
PIPE_SLOTS = 6                   # pipeline horizon in the global state
STATE_DIM = 1 + N_AGENTS * (3 + PIPE_SLOTS)

DEFAULTS = dict(
    horizon=50, max_order=100, holding_cost=0.5, backorder_cost=1.0,
    order_lead=2, order_lead_mfr=1, ship_lead=2, production_lead=2,
    init_inventory=12, demand_family="ar1", ar1_mu=12.0, ar1_rho=0.9, ar1_sigma=3.0,
    poisson_mu=8.0,
)


def ar1_step(prev_latent, mu, rho, sigma, rng):
    """One AR(1) step -> (demand_int, new_latent). Port of the validated sampler."""
    latent = mu + rho * (prev_latent - mu) + rng.normal(0.0, sigma)
    return max(0.0, float(round(latent))), latent


class _Pipeline:
    """Arrival-time-keyed queue: add(step_now, qty, lead) -> arrives at step_now+lead."""

    def __init__(self):
        self.q = {}

    def add(self, step_now, qty, lead):
        if qty > 0:
            k = step_now + int(lead)
            self.q[k] = self.q.get(k, 0.0) + float(qty)

    def receive(self, step_now):
        return float(self.q.pop(step_now, 0.0))

    def peek(self, step_now, slots):
        return [float(self.q.get(step_now + t, 0.0)) for t in range(1, slots + 1)]


class BeerGame:
    def __init__(self, config=None):
        self.cfg = {**DEFAULTS, **(config or {})}
        self.h = float(self.cfg["holding_cost"])
        self.b = float(self.cfg["backorder_cost"])
        self.horizon = int(self.cfg["horizon"])
        self.max_order = int(self.cfg["max_order"])

    # ------------------------------------------------------------------ lifecycle
    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.t = 0
        init = float(self.cfg["init_inventory"])
        self.inventory = {a: init for a in AGENTS}
        self.backlog = {a: 0.0 for a in AGENTS}
        self.on_order = {a: 0.0 for a in AGENTS}          # unfulfilled upstream orders
        self.last_incoming = {a: 0.0 for a in AGENTS}     # demand each stage saw last step
        self.ship_pipe = {a: _Pipeline() for a in AGENTS}
        self.order_pipe = {a: _Pipeline() for a in AGENTS}
        self._latent = float(self.cfg["ar1_mu"])          # AR(1) latent, init at mu
        self.last_demand = 0.0                            # last realized customer demand
        return self._obs()

    def _customer_demand(self):
        if self.cfg["demand_family"] == "ar1":
            d, self._latent = ar1_step(self._latent, float(self.cfg["ar1_mu"]),
                                       float(self.cfg["ar1_rho"]),
                                       float(self.cfg["ar1_sigma"]), self.rng)
            return d
        return float(self.rng.poisson(float(self.cfg["poisson_mu"])))

    # ------------------------------------------------------------------ dynamics
    def step(self, orders):
        """orders: array-like [N] of integer order quantities in [0, max_order].
        Returns (obs [N,4], local_costs [N], done, info)."""
        orders = np.clip(np.asarray(orders, dtype=float).round(), 0, self.max_order)

        # PHASE 1 -- receive shipments
        for a in AGENTS:
            got = self.ship_pipe[a].receive(self.t)
            self.inventory[a] += got
            self.on_order[a] -= got

        # PHASE 2 -- demand arrives, fulfill, ship downstream / produce
        for i, a in enumerate(AGENTS):
            if a == "retailer":
                demand = self._customer_demand()
                self.last_demand = demand
            else:
                demand = self.order_pipe[AGENTS[i - 1]].receive(self.t)
            if a == "manufacturer":
                # production: the manufacturer's OWN orders, arriving up its order
                # pipe (lead 1), become production into its own shipment pipe
                # (production lead 2) -- total replenishment lead 3.
                requests = self.order_pipe[a].receive(self.t)
                if requests > 0:
                    self.ship_pipe[a].add(self.t, requests, self.cfg["production_lead"])
            self.last_incoming[a] = demand
            total_req = demand + self.backlog[a]
            fulfilled = min(self.inventory[a], total_req)
            self.inventory[a] -= fulfilled
            self.backlog[a] = total_req - fulfilled
            if a != "retailer" and fulfilled > 0:   # ship down to the stage below
                self.ship_pipe[AGENTS[i - 1]].add(self.t, fulfilled, self.cfg["ship_lead"])

        # PHASE 3 -- place orders upstream
        for i, a in enumerate(AGENTS):
            o = float(orders[i])
            if o > 0:
                self.on_order[a] += o
                lead = self.cfg["order_lead_mfr"] if a == "manufacturer" else self.cfg["order_lead"]
                self.order_pipe[a].add(self.t, o, lead)

        # PHASE 4 -- costs
        costs = np.array([self.h * self.inventory[a] + self.b * self.backlog[a]
                          for a in AGENTS], dtype=np.float32)

        self.t += 1
        done = self.t >= self.horizon
        return self._obs(), costs, done, {"demand": self.last_demand, "t": self.t}

    # ------------------------------------------------------------------ views
    def _obs(self):
        return np.array([[self.inventory[a], self.backlog[a], self.on_order[a],
                          self.last_incoming[a]] for a in AGENTS], dtype=np.float32)

    def global_state(self):
        s = [float(self.t)]
        for a in AGENTS:
            s.extend([self.inventory[a], self.backlog[a], self.on_order[a]])
            s.extend(self.ship_pipe[a].peek(self.t, PIPE_SLOTS))
        return np.array(s, dtype=np.float32)

    @staticmethod
    def inventory_position(obs_row):
        """IP = inventory - backlog + on_order, from one raw observation row."""
        return float(obs_row[0]) - float(obs_row[1]) + float(obs_row[2])
