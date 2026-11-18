"""Actor-Critic-Netz für kontinuierliche Aktionen (Lenken, Gas/Bremse).

Der Policy-Kopf gibt eine Gauss-Verteilung über tanh-gedämpfte
Aktionen in [-1, 1] aus (Standard-Ansatz für PPO mit Begrenzungen).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

LOG_STD_MIN = -3.0
LOG_STD_MAX = 0.0
EPS = 1e-6


def _ortho_init(module: nn.Module, gain: float) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain)
        nn.init.zeros_(module.bias)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: tuple[int, ...] = (128, 128)):
        super().__init__()
        self.act_dim = act_dim

        layers: list[nn.Module] = []
        d = obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.Tanh()]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.mu = nn.Linear(d, act_dim)
        self.v = nn.Linear(d, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

        self.trunk.apply(lambda m: _ortho_init(m, np.sqrt(2)))
        _ortho_init(self.mu, 0.01)
        _ortho_init(self.v, 1.0)

    # ------------------------------------------------------------------
    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.trunk(obs)

    def _dist(self, obs: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        feat = self._features(obs)
        mu = self.mu(feat)
        std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp()
        dist = Normal(mu, std.expand_as(mu))
        return dist, self.v(feat).squeeze(-1)

    @staticmethod
    def _squash_log_prob(dist: Normal, pre_tanh: torch.Tensor,
                         action: torch.Tensor) -> torch.Tensor:
        logp = dist.log_prob(pre_tanh).sum(-1)
        logp -= torch.log(1.0 - action.pow(2) + EPS).sum(-1)
        return logp

    # ------------------------------------------------------------------
    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False):
        """Aktionen samplen; Rückgabe (action, logp, value)."""
        dist, value = self._dist(obs)
        if deterministic:
            pre_tanh = dist.mean
        else:
            pre_tanh = dist.sample()
        action = torch.tanh(pre_tanh)
        logp = self._squash_log_prob(dist, pre_tanh, action)
        return action, logp, value

    @torch.no_grad()
    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.v(self._features(obs)).squeeze(-1)

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        """logp, entropy (Approximation) und value für gegebene Paare."""
        dist, value = self._dist(obs)
        pre_tanh = torch.atanh(action.clamp(-0.999999, 0.999999))
        logp = self._squash_log_prob(dist, pre_tanh, action)
        entropy = dist.entropy().sum(-1)
        return logp, entropy, value
