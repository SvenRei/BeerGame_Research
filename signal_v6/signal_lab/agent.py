"""signal_lab/agent.py -- the whole trainable surface: five pieces.

SharedActor : trunk (ReLU x2, 256) -> GRU(256) -> {action head over the S-grid,
              message head}. ONE network shared by all four echelons; role identity
              enters as a one-hot. The message head ALWAYS exists and is ALWAYS
              evaluated (so parameter counts, gradient graphs, and RNG consumption
              are identical across every arm -- T-PARAM / T-SYM); it only trains when
              content='learned' because only then does its output reach the loss.
Critic      : centralized, global_state/100 -> ReLU(256) x2 -> N values (per-agent,
              needed for the beta axis; redundant at beta=1 and that redundancy is the
              price of one architecture across the whole axis).

Action space: order-up-to level S on a categorical grid linspace(0, s_max=100, 41)
(registered decision R1: tighter than the legacy [0,160] to kill the cold-start
flood). The executed order is clip(round(S - IP), 0, max_order), computed from the
agent's OWN observation -- base-stock semantics, comparable to the base-stock
baselines by construction.

Learned messages: m = 100 * tanh(msg_head(.)) so every content lives in the same
bounded demand-unit space and the actor input rule is uniform: everything / 100.

Not a single Tanh in any value path; ReLU everywhere (reference design, R6).
"""
import numpy as np
import torch
import torch.nn as nn

from env.beer_game import N_AGENTS, OBS_DIM, STATE_DIM, BeerGame


def s_grid(bins=41, s_max=100.0):
    return torch.linspace(0.0, float(s_max), int(bins))


def orders_from_s(s_values, obs_raw, max_order):
    """order_i = clip(round(S_i - IP_i), 0, max_order); obs_raw [N, OBS_DIM] unscaled."""
    ip = np.array([BeerGame.inventory_position(obs_raw[i]) for i in range(len(obs_raw))])
    return np.clip(np.round(np.asarray(s_values) - ip), 0, max_order).astype(int)


def _ortho(layer, gain):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class SharedActor(nn.Module):
    def __init__(self, msg_dim, hidden=256, bins=41, s_max=100.0):
        super().__init__()
        self.M, self.H, self.bins = int(msg_dim), int(hidden), int(bins)
        in_dim = OBS_DIM + N_AGENTS + self.M          # obs/100 || role one-hot || msg/100
        self.fc1 = _ortho(nn.Linear(in_dim, hidden), np.sqrt(2))
        self.fc2 = _ortho(nn.Linear(hidden, hidden), np.sqrt(2))
        self.gru = nn.GRU(hidden, hidden)
        self.action_head = _ortho(nn.Linear(hidden, bins), 0.01)   # near-uniform cold start
        self.msg_head = _ortho(nn.Linear(hidden, self.M), 0.01)
        self.register_buffer("grid", s_grid(bins, s_max))
        self.role = torch.eye(N_AGENTS)

    def _inp(self, obs_t, msg_t):
        """obs_t [..., N, OBS_DIM] raw, msg_t [..., N, M] demand units -> input."""
        role = self.role.to(obs_t.device).expand(*obs_t.shape[:-1], N_AGENTS)
        return torch.cat([obs_t / 100.0, role, msg_t / 100.0], dim=-1)

    def cell(self, obs_t, msg_t, h):
        """One step for all N agents. -> (logits [N,bins], m_out [N,M], h')"""
        x = torch.relu(self.fc2(torch.relu(self.fc1(self._inp(obs_t, msg_t)))))
        out, h = self.gru(x.unsqueeze(0), h)
        z = out.squeeze(0)
        return self.action_head(z), 100.0 * torch.tanh(self.msg_head(z)), h

    def init_hidden(self, device="cpu"):
        return torch.zeros(1, N_AGENTS, self.H, device=device)

    def message(self, obs_t, h):
        """Sender message from (obs_t, h_{t-1}) -- computed BEFORE the incoming message
        exists (breaks the same-step circularity; identical rule in rollout and update)."""
        x = torch.relu(self.fc2(torch.relu(self.fc1(
            self._inp(obs_t, torch.zeros(obs_t.shape[0], self.M,
                                         device=obs_t.device))))))
        z = (self.gru(x.unsqueeze(0), h)[0]).squeeze(0)
        return 100.0 * torch.tanh(self.msg_head(z))


class Critic(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            _ortho(nn.Linear(STATE_DIM, hidden), np.sqrt(2)), nn.ReLU(),
            _ortho(nn.Linear(hidden, hidden), np.sqrt(2)), nn.ReLU(),
            _ortho(nn.Linear(hidden, N_AGENTS), 1.0))

    def forward(self, state):                        # raw global state, any leading dims
        return self.net(state / 100.0)


def count_params(module):
    return sum(p.numel() for p in module.parameters())
