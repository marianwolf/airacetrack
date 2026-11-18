"""PPO (Proximal Policy Optimization) – kompakte CPU-Variante.

Enthält:
* Beobachtungs-Normalisierung (laufender Mittelwert/Varianz)
* GAE(λ)-Vorteilsschätzung
* Clipped-Surrogate-Update mit Entropie-Bonus
* Speichern/Laden eines Trainings-Checkpoints
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .policy import ActorCritic


class RunningMeanStd:
    """Laufender Mittelwert/Varianz nach Welford (vektorisiert)."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        x = np.atleast_2d(x)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / np.sqrt(self.var + 1e-8)).clip(-5.0, 5.0)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d: dict) -> None:
        self.mean = np.asarray(d["mean"], dtype=np.float64)
        self.var = np.asarray(d["var"], dtype=np.float64)
        self.count = float(d["count"])


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    minibatches: int = 4
    ent_coef: float = 0.02         # Entropie-Bonus (Exploration)
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr_decay_steps: int = 0      # 0 = konstante Lernrate


class PPO:
    def __init__(self, obs_dim: int, act_dim: int, config: PPOConfig | None = None,
                 hidden: tuple[int, ...] = (128, 128), device: str = "cpu") -> None:
        self.cfg = config or PPOConfig()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden = tuple(hidden)
        self.device = torch.device(device)

        self.model = ActorCritic(obs_dim, act_dim, self.hidden).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr,
                                          eps=1e-5)
        self.obs_rms = RunningMeanStd((obs_dim,))
        self.updates = 0

    # ------------------------------------------------------------------
    def _lr(self) -> float:
        if self.cfg.lr_decay_steps <= 0:
            return self.cfg.lr
        frac = min(1.0, self.updates / max(self.cfg.lr_decay_steps, 1))
        return self.cfg.lr * (1.0 - frac)

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        return self.obs_rms.normalize(obs)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False):
        """Roh-Beobachtungen; Rückgabe (actions, logps, values, normierte obs).

        Die Normalisierung wird mit demselben laufenden Mittelwert/Varianz
        angewendet, den auch die Netzwerkbewertung sieht – die normalisierten
        Beobachtungen müssen zusammen mit den logp-Werten im Buffer landen,
        sonst stimmen die PPO-Verhältnisse bei der Aktualisierung nicht.
        """
        self.model.eval()
        self.obs_rms.update(obs)
        norm = self.obs_rms.normalize(obs)
        o = torch.as_tensor(norm, dtype=torch.float32, device=self.device)
        actions, logp, values = self.model.act(o, deterministic)
        return (actions.cpu().numpy(), logp.cpu().numpy(), values.cpu().numpy(),
                norm.astype(np.float32))

    @torch.no_grad()
    def value(self, obs: np.ndarray) -> np.ndarray:
        o = torch.as_tensor(self.normalize(obs), dtype=torch.float32,
                            device=self.device)
        return self.model.value(o).cpu().numpy()

    # ------------------------------------------------------------------
    @staticmethod
    def compute_returns(rewards: np.ndarray, values: np.ndarray,
                        next_values: np.ndarray, dones: np.ndarray,
                        gamma: float, gae_lambda: float
                        ) -> tuple[np.ndarray, np.ndarray]:
        """GAE(λ). Alle Arrays haben Form (T, N).

        next_values[t] enthält bereits γ·V(s_{t+1})-Ersatz: für abgeschlossene
        Episoden den Terminalwert (0), bei Abbruch durch Zeitlimit den Wert des
        letzten echten Zustands (Bootstrapping).
        """
        T, N = rewards.shape
        advantages = np.zeros((T, N), dtype=np.float32)
        last = np.zeros(N, dtype=np.float32)
        for t in range(T - 1, -1, -1):
            delta = rewards[t] + next_values[t] - values[t]
            last = delta + gamma * gae_lambda * (1.0 - dones[t]) * last
            advantages[t] = last
        returns = advantages + values
        return advantages, returns

    # ------------------------------------------------------------------
    def update(self, batch: dict) -> dict:
        """Ein PPO-Update über die gesammelte Rollout-Batch."""
        cfg = self.cfg
        self.model.train()
        for g in self.optimizer.param_groups:
            g["lr"] = self._lr()

        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32,
                                  device=self.device)
        old_logp = torch.as_tensor(batch["logps"], dtype=torch.float32,
                                   device=self.device)
        adv = torch.as_tensor(batch["advantages"], dtype=torch.float32,
                              device=self.device)
        ret = torch.as_tensor(batch["returns"], dtype=torch.float32,
                              device=self.device)

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = obs.shape[0]
        mb = max(n // cfg.minibatches, 1)
        stats = {"pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "clip_frac": 0.0}
        count = 0

        for _ in range(cfg.epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, mb):
                idx = perm[start : start + mb]
                if idx.numel() == 0:
                    continue
                logp, entropy, value = self.model.evaluate(obs[idx], actions[idx])

                ratio = torch.exp(logp - old_logp[idx])
                m1 = ratio * adv[idx]
                m2 = torch.clamp(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip) * adv[idx]
                pi_loss = -torch.min(m1, m2).mean()

                v_loss = nn.functional.mse_loss(value, ret[idx])
                entropy_loss = -entropy.mean()

                loss = pi_loss + cfg.vf_coef * v_loss + cfg.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    clip_frac = ((ratio - 1.0).abs() > cfg.clip).float().mean()
                stats["pi_loss"] += pi_loss.item()
                stats["v_loss"] += v_loss.item()
                stats["entropy"] += entropy.mean().item()
                stats["clip_frac"] += clip_frac.item()
                count += 1

        self.updates += 1
        if count:
            for k in stats:
                stats[k] /= count
        return stats

    # ------------------------------------------------------------------
    def save(self, path: str, extra: dict | None = None) -> None:
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "obs_rms": self.obs_rms.state_dict(),
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "hidden": list(self.hidden),
            "ppo_config": self.cfg.__dict__,
            "updates": self.updates,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> tuple["PPO", dict]:
        payload = torch.load(path, map_location=device, weights_only=False)
        cfg = PPOConfig(**payload.get("ppo_config", {}))
        agent = cls(payload["obs_dim"], payload["act_dim"], cfg,
                    hidden=tuple(payload.get("hidden", (128, 128))), device=device)
        agent.model.load_state_dict(payload["model"])
        agent.optimizer.load_state_dict(payload["optimizer"])
        agent.obs_rms.load_state_dict(payload["obs_rms"])
        agent.updates = payload.get("updates", 0)
        return agent, payload
