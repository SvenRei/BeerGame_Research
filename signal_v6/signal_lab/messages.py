"""signal_lab/messages.py -- the treatment. The ONLY thing that varies across arms.

MessageProvider maps environment state -> incoming messages [N, M] per step. Every
content produces a per-sender vector; a topology routing matrix R decides who receives
whom; the actor sees only "a vector arrived". M is FIXED across the whole study so the
actor architecture is identical in every arm (nocomm = zeros, not a different network).

Content ladder (V(content)):
  nocomm   zeros                                   -- the control
  raw      sender's last observed incoming order   -- for the retailer that is customer
                                                      demand d_{t-1}
  ip       sender's inventory position
  arpred   analytic AR(1) conditional mean mu + rho (d_{t-1} - mu)
  dhatc    frozen forecaster's one-step prediction -- no gradient, ever
  learned  the actor's message head output         -- the only trainable content; the
           gradient flows sender->receiver through the differentiable channel (DIAL)
           during the update's in-graph recompute

Interventions (do(m), evaluation only) are wrappers around a provider: honest /
zeroed / shuffled / cross. `zeroed` IS the nocomm provider -- one implementation,
two names, by design.

At t=0 no demand has been observed; value-type contents emit the neutral prior mu
(documented choice, applied uniformly to raw/arpred/dhatc).
"""
import numpy as np
import torch
import torch.nn as nn

from env.beer_game import AGENTS, N_AGENTS, BeerGame

CONTENTS = ("nocomm", "raw", "ip", "arpred", "dhatc", "learned")
TOPOLOGIES = ("retailer_broadcast", "neighbor")
INTERVENTIONS = ("honest", "zeroed", "shuffled", "cross")


def routing_matrix(topology):
    """R[i, j] = 1 iff agent i receives agent j's message."""
    R = np.zeros((N_AGENTS, N_AGENTS), dtype=np.float32)
    if topology == "retailer_broadcast":
        R[1:, 0] = 1.0                       # everyone upstream hears the retailer
    elif topology == "neighbor":
        for i in range(1, N_AGENTS):
            R[i, i - 1] = 1.0                # each stage hears its downstream neighbor
    else:
        raise ValueError(f"unknown topology {topology!r} (choose from {TOPOLOGIES})")
    return R


class FrozenForecaster(nn.Module):
    """Tiny GRU one-step demand predictor. Frozen at load: requires_grad=False on every
    parameter, so no gradient can ever reach it (T-GRAD-2)."""

    def __init__(self, hidden=16):
        super().__init__()
        self.gru = nn.GRU(1, hidden)
        self.head = nn.Linear(hidden, 1)
        self.hidden = hidden

    def forward(self, d_prev, h):            # d_prev [N,1] in demand/100 units
        out, h = self.gru(d_prev.unsqueeze(0), h)
        return self.head(out.squeeze(0)), h

    @classmethod
    def load_frozen(cls, path, device="cpu"):
        payload = torch.load(path, map_location=device, weights_only=False)
        m = cls(hidden=int(payload.get("hidden", 16)))
        m.load_state_dict(payload["state_dict"])
        for p in m.parameters():
            p.requires_grad_(False)
        m.eval()
        return m


class MessageProvider:
    def __init__(self, content, topology, msg_dim, cfg=None, forecaster_path=None,
                 device="cpu"):
        if content not in CONTENTS:
            raise ValueError(f"unknown content {content!r} (choose from {CONTENTS})")
        self.content = content
        self.M = int(msg_dim)
        self.R = routing_matrix(topology)
        self.cfg = dict(cfg or {})
        self.mu = float(self.cfg.get("ar1_mu", 12.0))
        self.rho = float(self.cfg.get("ar1_rho", 0.9))
        self.device = device
        self.forecaster = None
        if content == "dhatc":
            if not forecaster_path:
                raise ValueError("content=dhatc requires forecaster_path (fail-closed: "
                                 "no silent fallback to another content)")
            self.forecaster = FrozenForecaster.load_frozen(forecaster_path, device)
        self.reset()

    def reset(self):
        self._fh = None
        self._seen_step = False              # False until the first env step

    # ------------------------------------------------------------------ core
    def _sender_values(self, env: BeerGame, obs):
        """Per-sender scalar in demand units, before routing. [N]"""
        v = np.zeros(N_AGENTS, dtype=np.float32)
        if self.content == "raw":
            for i, a in enumerate(AGENTS):
                v[i] = env.last_incoming[a] if self._seen_step else self.mu
        elif self.content == "ip":
            for i in range(N_AGENTS):
                v[i] = BeerGame.inventory_position(obs[i])
        elif self.content == "arpred":
            for i, a in enumerate(AGENTS):
                d = env.last_incoming[a] if self._seen_step else self.mu
                v[i] = self.mu + self.rho * (d - self.mu)
        elif self.content == "dhatc":
            d = np.array([[env.last_incoming[a] if self._seen_step else self.mu]
                          for a in AGENTS], dtype=np.float32) / 100.0
            with torch.no_grad():
                if self._fh is None:
                    self._fh = torch.zeros(1, N_AGENTS, self.forecaster.hidden,
                                           device=self.device)
                pred, self._fh = self.forecaster(torch.tensor(d, device=self.device),
                                                 self._fh)
            v[:] = pred.squeeze(-1).cpu().numpy() * 100.0
        return v

    def incoming(self, env, obs, learned_msgs=None):
        """-> incoming messages [N, M] in demand units. Called once per env step,
        BEFORE actions. `learned_msgs` [N, M] must be given iff content='learned'."""
        if self.content == "nocomm":
            self._seen_step = True
            return np.zeros((N_AGENTS, self.M), dtype=np.float32)
        if self.content == "learned":
            if learned_msgs is None:
                raise ValueError("content=learned requires learned_msgs from the actor")
            out = self.R @ np.asarray(learned_msgs, dtype=np.float32)
        else:
            m = np.zeros((N_AGENTS, self.M), dtype=np.float32)
            m[:, 0] = self._sender_values(env, obs)         # scalar in slot 0, zero-pad
            out = self.R @ m
        self._seen_step = True
        return out

    def route_in_graph(self, learned_msgs_t):
        """Torch routing for the update's in-graph recompute (content='learned' only):
        keeps the sender->receiver gradient path alive. learned_msgs_t [N, M]."""
        R = torch.tensor(self.R, dtype=learned_msgs_t.dtype,
                         device=learned_msgs_t.device)
        return R @ learned_msgs_t


# ---------------------------------------------------------------------- interventions
class InterventionWrapper:
    """do(m): manipulate the channel at evaluation time. Complete by construction --
    the channel is the only treatment surface, so there is no path around this."""

    def __init__(self, provider, mode, seed=0):
        if mode not in INTERVENTIONS:
            raise ValueError(f"unknown intervention {mode!r} (choose from {INTERVENTIONS})")
        self.p = provider
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.content = provider.content
        self.M = provider.M

    def reset(self):
        self.p.reset()

    def incoming(self, env, obs, learned_msgs=None):
        m = self.p.incoming(env, obs, learned_msgs)
        if self.mode == "honest":
            return m
        if self.mode == "zeroed":
            return np.zeros_like(m)
        if self.mode == "shuffled":                 # permute receivers each step
            return m[self.rng.permutation(len(m))]
        if self.mode == "cross":                    # rotate: everyone gets a wrong line
            return np.roll(m, 1, axis=0)

    def route_in_graph(self, learned_msgs_t):
        return self.p.route_in_graph(learned_msgs_t)
