"""Trainiertes Modell auswerten:

    python -m airacetrack.evaluate runs/.../best.pt                 # Statistik
    python -m airacetrack.evaluate runs/.../best.pt --gif lauf.gif  # Animation
    python -m airacetrack.evaluate runs/.../best.pt --plot traj.png # Fahrtrajektorie
    python -m airacetrack.evaluate runs/.../best.pt --live          # live ansehen
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from .env import EnvConfig, RaceEnv
from .ppo import PPO
from .track import Track
from .tracks import CONTROL_POINTS, TRACK_CONFIG


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trainierte KI auswerten")
    p.add_argument("checkpoint", help="Pfad zu .pt (best.pt / latest.pt)")
    p.add_argument("--episodes", type=int, default=3, help="Anzahl Episoden")
    p.add_argument("--laps", type=int, default=2,
                   help="Episodenende nach N Umläufen (0 = Zeitlimit)")
    p.add_argument("--stochastic", action="store_true",
                   help="Aktionen samplen statt deterministisch")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--gif", type=str, default=None, help="Animation als GIF speichern")
    p.add_argument("--plot", type=str, default=None, help="Fahrtrajektorie als PNG")
    p.add_argument("--live", action="store_true", help="Live mit Matplotlib ansehen")
    return p.parse_args()


# ----------------------------------------------------------------------
def predict(agent: PPO, obs: np.ndarray, deterministic: bool) -> np.ndarray:
    """Aktion ohne Veränderung der Normalisierungs-Statistik."""
    import torch

    with torch.no_grad():
        o = torch.as_tensor(agent.normalize(obs[None]), dtype=torch.float32)
        action, _, _ = agent.model.act(o, deterministic=deterministic)
    return action.cpu().numpy()[0]


def run_episode(agent: PPO, env: RaceEnv, cfg: EnvConfig, deterministic: bool,
                stop_after_laps: int, record_every: int = 0):
    """Eine Episode fahren. Rückgabe: info, Trajektorie, Frames(Indices)."""
    obs = env.reset()
    traj: list[tuple[float, float, float, bool]] = []
    frames: list[int] = []
    done, step = False, 0
    while not done:
        action = predict(agent, obs, deterministic)
        obs, _, done, info = env.step(action)
        traj.append((env.car.x, env.car.y, env.car.speed, info["on_grass"]))
        if record_every and step % record_every == 0:
            frames.append(step)
        step += 1
        if stop_after_laps and info["lap"] >= stop_after_laps:
            done = True
    return info, traj, frames


# ----------------------------------------------------------------------
class Renderer:
    """Zeichnet einen Simulations-Schritt in eine Matplotlib-Figur."""

    def __init__(self, track: Track, cfg: EnvConfig, figsize=(9, 8)):
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.patches import Polygon

        self.track, self.cfg = track, cfg
        self.fig, self.ax = plt.subplots(figsize=figsize)
        track.plot(self.ax)
        self.ax.set_aspect("equal")
        self.car_patch = Polygon(np.zeros((3, 2)), closed=True, color="#e10600",
                                 zorder=6, linewidth=0)
        self.ax.add_patch(self.car_patch)
        self.rays = LineCollection(np.zeros((cfg.n_rays, 2, 2)),
                                   colors="orange", alpha=0.55, linewidths=1.0,
                                   zorder=5)
        self.ax.add_collection(self.rays)
        self.hud = self.ax.text(
            0.02, 0.98, "", transform=self.ax.transAxes, va="top", fontsize=11,
            family="monospace",
            bbox=dict(boxstyle="round", fc="white", alpha=0.75),
        )
        all_pts = np.vstack([track.left, track.right])
        pad = 60.0
        self.ax.set_xlim(all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
        self.ax.set_ylim(all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)

    def update(self, env: RaceEnv, info: dict) -> None:
        car = env.car
        c, s = np.cos(car.heading), np.sin(car.heading)
        # Fahrzeug-Dreieck (vorn +2.3, hinten -2.2, halbe Breite 0.95)
        local = np.array([[2.3, 0.0], [-2.2, 0.95], [-2.2, -0.95]])
        rot = np.array([[c, -s], [s, c]])
        pts = local @ rot.T + car.position
        self.car_patch.set_xy(pts)

        angles = env._ray_angles + car.heading
        d = self.track.raycast(car.position, angles, self.cfg.ray_max)
        segs = np.stack([np.repeat(car.position[None], self.cfg.n_rays, axis=0),
                         car.position + np.stack([np.cos(angles), np.sin(angles)],
                                                 axis=1) * d[:, None]], axis=1)
        self.rays.set_segments(segs)

        self.hud.set_text(
            f"{car.speed*3.6:6.1f} km/h   Runde {info['lap']}\n"
            f"Zeit   {info['time']:6.1f} s\n"
            f"Fortschritt {info['progress']:7.0f} m"
            + ("   [GRAS!]" if info["on_grass"] else "")
        )
        self.fig.canvas.draw()

    def close(self) -> None:
        import matplotlib.pyplot as plt
        plt.close(self.fig)


# ----------------------------------------------------------------------
def save_gif(agent: PPO, track: Track, cfg: EnvConfig, deterministic: bool,
             path: str, laps: int, seed: int) -> None:
    from matplotlib.animation import PillowWriter

    env = RaceEnv(track, cfg, seed=seed)
    renderer = Renderer(track, cfg)
    obs = env.reset()
    info = {"lap": 0, "time": 0.0, "progress": 0.0, "on_grass": False}
    done = False
    step = 0
    max_steps = 8000 if not laps else 20_000

    writer = PillowWriter(fps=10)
    with writer.saving(renderer.fig, path, dpi=90):
        while not done and step < max_steps:
            action = predict(agent, obs, deterministic)
            obs, _, done, info = env.step(action)
            if laps and info["lap"] >= laps:
                done = True
            renderer.update(env, info)
            if step % 2 == 0:          # 2 × 0,05 s → 10 fps in Echtzeit
                writer.grab_frame()
            step += 1
    renderer.close()
    print(f"GIF gespeichert: {path} ({step} Schritte, "
          f"{'Unfall' if info['crashed'] else 'Ziel'})")


def save_trajectory(track: Track, traj, path: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    pts = np.array([[p[0], p[1]] for p in traj])
    speeds = np.array([p[2] for p in traj])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap="turbo", linewidths=2.0)
    lc.set_array((speeds[:-1] + speeds[1:]) / 2)

    fig, ax = plt.subplots(figsize=(10, 9))
    track.plot(ax)
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect("equal")
    cbar = fig.colorbar(lc, ax=ax, shrink=0.8)
    cbar.set_label("Geschwindigkeit [m/s]")
    ax.set_title(f"Fahrtrajektorie – {len(traj)} Schritte, "
                 f"Ø {speeds.mean()*3.6:.0f} km/h, max {speeds.max()*3.6:.0f} km/h")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Trajektorie gespeichert: {path}")


# ----------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    track = Track(CONTROL_POINTS, **TRACK_CONFIG)
    cfg = EnvConfig()

    agent, payload = PPO.load(args.checkpoint)
    print(f"Modell geladen: {args.checkpoint}")
    print(f"  trainiert für {payload.get('train_steps', '?')} Schritte, "
          f"Update #{agent.updates}")

    if args.gif:
        save_gif(agent, track, cfg, not args.stochastic, args.gif,
                 args.laps, args.seed)
        return

    infos = []
    traj = None
    t0 = time.time()
    for ep in range(args.episodes):
        env = RaceEnv(track, cfg, seed=args.seed + ep)
        info, traj_ep, _ = run_episode(agent, env, cfg, not args.stochastic,
                                       args.laps)
        infos.append(info)
        traj = traj_ep
        lap_txt = (f"{info['lap']} Umläufe" if not args.laps
                   else f"Rundenzeiten bis Umlauf {info['lap']}")
        print(f"Episode {ep+1}: {'Unfall' if info['crashed'] else 'ok':>8} | "
              f"Fortschritt {info['progress']:7.0f} m | "
              f"Ø {info['progress']/max(info['time'],1e-9):5.1f} m/s | "
              f"bis {info['lap']} Umläufe in {info['time']:5.1f}s")
    dt = time.time() - t0

    crashed = sum(i["crashed"] for i in infos)
    print(f"\nZusammenfassung: {len(infos)} Episoden, {crashed} Unfälle, "
          f"{dt:.1f}s Echtzeit")
    mean_progress = np.mean([i["progress"] for i in infos])
    print(f"Ø Fortschritt: {mean_progress:.0f} m von {track.length:.0f} m pro "
          f"Episodenstart-Zeitfenster")

    if args.plot:
        save_trajectory(track, traj, args.plot)
    if args.live:
        live(agent, track, cfg, not args.stochastic, args.seed)


def live(agent: PPO, track: Track, cfg: EnvConfig, deterministic: bool,
         seed: int) -> None:
    import matplotlib.pyplot as plt

    env = RaceEnv(track, cfg, seed=seed)
    renderer = Renderer(track, cfg)
    plt.ion()
    renderer.fig.show()
    obs = env.reset()
    done = False
    while plt.fignum_exists(renderer.fig.number):
        if not done:
            action = predict(agent, obs, deterministic)
            obs, _, done, info = env.step(action)
            renderer.update(env, info)
        plt.pause(0.001)
        if done:
            time.sleep(1.0)
            obs = env.reset()
            done = False
    renderer.close()


if __name__ == "__main__":
    main()
