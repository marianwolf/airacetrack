"""Training starten:

    python -m airacetrack.train --steps 500000

Speichert Checkpoints und eine CSV-Protokolldatei unter `runs/<zeitstempel>/`.
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import deque
from pathlib import Path

import numpy as np

from .env import ACT_DIM, OBS_DIM, EnvConfig, RaceEnv
from .ppo import PPO, PPOConfig
from .track import Track
from .tracks import CONTROL_POINTS, TRACK_CONFIG


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KI auf der Rennstrecke trainieren (PPO)")
    p.add_argument("--steps", type=int, default=500_000,
                   help="Gesamtzahl Trainings-Schritte (Aktionen)")
    p.add_argument("--num-envs", type=int, default=32, help="Parallele Umgebungen")
    p.add_argument("--rollout", type=int, default=128,
                   help="Schritte pro Umgebung und Rollout")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--total-laps", type=int, default=0,
                   help="Episodenende nach N Umläufen (0 = nur Zeitlimit)")
    p.add_argument("--resume", type=str, default=None,
                   help="Checkpoint fortsetzen (.pt)")
    p.add_argument("--out", type=str, default="runs", help="Ausgabeordner")
    p.add_argument("--log-every", type=int, default=10_000,
                   help="Alle N Schritte Protokoll ausgeben")
    p.add_argument("--save-every", type=int, default=50_000,
                   help="Alle N Schritten Checkpoint speichern")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # Umgebungen
    track = Track(CONTROL_POINTS, **TRACK_CONFIG)
    env_cfg = EnvConfig()
    envs = [RaceEnv(track, env_cfg, seed=args.seed + i) for i in range(args.num_envs)]
    obs_dim, act_dim = envs[0].observation_dim, envs[0].action_dim
    assert (obs_dim, act_dim) == (OBS_DIM, ACT_DIM)

    # Agent
    ppo_cfg = PPOConfig(
        lr=args.lr,
        gamma=args.gamma,
        # Lineare Lernrate-Zerrung bis zum Ende der Laufzeit (0 = konstant)
        lr_decay_steps=max(args.steps // (args.rollout * args.num_envs), 1),
    )
    if args.resume:
        agent, payload = PPO.load(args.resume)
        if agent.obs_dim != obs_dim or agent.act_dim != act_dim:
            raise SystemExit(
                f"Checkpoint wurde mit {agent.obs_dim} Beobachtungen/"
                f"{agent.act_dim} Aktionen trainiert, die aktuelle Umgebung "
                f"hat {obs_dim}/{act_dim} – Abbruch."
            )
        print(f"Fortgesetzt von {args.resume} "
              f"(Update #{agent.updates}, {payload.get('train_steps', 0)} Schritte)")
    else:
        agent = PPO(obs_dim, act_dim, ppo_cfg)

    # Ausgabeordner
    run_dir = Path(args.out) / time.strftime("%Y%m%d-%H%M%S")
    if args.resume and Path(args.resume).parent.exists():
        run_dir = Path(args.resume).parent
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.csv"
    if not log_path.exists():
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", newline="") as f:
                csv.writer(f).writerow(
                    ["global_step", "episodes", "mean_reward", "mean_progress",
                     "crash_rate", "stranded_rate", "mean_speed", "mean_lap_time",
                     "updates", "fps"]
                )
        except OSError:
            pass  # safe_log meldet den Fehler später mit Warnung

    print(f"Strecke: {track.length:.0f} m | {len(track.checkpoint_idx)} Checkpoints | "
          f"Env-Config: dt={env_cfg.dt}s, max_steps={env_cfg.max_steps}")
    print(f"Netz: {OBS_DIM} Beobachtungen → {ACT_DIM} Aktionen | "
          f"{args.num_envs} Umgebungen × {args.rollout} Schritte/Rollout")
    print(f"Ausgabe: {run_dir}\n")

    warn_state = {"log": False, "save": False}

    def safe_log(row) -> None:
        """CSV-Zeile anhängen – wiederholt angelegtes Verzeichnis, kein Abbruch."""
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", newline="") as f:
                csv.writer(f).writerow(row)
        except OSError as e:
            if not warn_state["log"]:
                warn_state["log"] = True
                print(f"\nWARNUNG: Protokoll nicht schreibbar ({e}) – "
                      f"Training läuft weiter ohne log.csv.")

    def safe_save(extra: dict, name: str = "latest.pt") -> None:
        """Checkpoint speichern – Ordner neu anlegen, Fehler nicht töten."""
        path = run_dir / name
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            agent.save(str(path), extra)
        except OSError as e:
            if not warn_state["save"]:
                warn_state["save"] = True
                print(f"\nWARNUNG: Checkpoint {path} nicht speicherbar ({e}) – "
                      f"Training läuft weiter ohne Checkpoints.")

    # Rollout-Buffers
    T, N = args.rollout, args.num_envs
    buf_obs = np.zeros((T, N, obs_dim), dtype=np.float32)
    buf_act = np.zeros((T, N, act_dim), dtype=np.float32)
    buf_logp = np.zeros((T, N), dtype=np.float32)
    buf_val = np.zeros((T, N), dtype=np.float32)
    buf_rew = np.zeros((T, N), dtype=np.float32)
    buf_done = np.zeros((T, N), dtype=np.float32)
    buf_nextval = np.zeros((T, N), dtype=np.float32)

    obs = np.stack([e.reset() for e in envs])
    ep_reward = np.zeros(N)
    ep_progress = np.zeros(N)
    ep_speed_sum = np.zeros(N)
    ep_steps = np.zeros(N)
    recent = deque(maxlen=50)          # letzte Episoden
    lap_times: list[float] = []
    global_step = 0
    episodes = 0
    best_score = -np.inf
    t_start = time.time()
    t_log = t_start
    steps_log = 0

    print(f"{'Schritt':>10} {'Episoden':>9} {'Rewards':>8} {'Fortschritt':>12} "
          f"{'Crash%':>7} {'Strand%':>8} {'Tempo':>7} {'Rundenzeit':>11} {'FPS':>6}")

    while global_step < args.steps:
        # ---------------- Rollout sammeln ----------------
        for t in range(T):
            actions, logps, values, norm_obs = agent.act(obs)
            buf_obs[t] = norm_obs          # normalisiert wie bei act()/logps!
            buf_act[t] = actions
            buf_logp[t] = logps
            buf_val[t] = values

            next_obs = np.empty_like(obs)
            # (Index, Zustand, aus dem weitergerechnet wird) – Unfall-Enden fehlen
            # bewusst (= 0), Zeitlimit-Enden stehen mit dem Endzustand dabei.
            pending: list[tuple[int, np.ndarray]] = []
            for i, env in enumerate(envs):
                o2, r, done, info = env.step(actions[i])
                buf_rew[t, i] = r
                ep_reward[i] += r
                ep_progress[i] = info["progress"]
                ep_speed_sum[i] += info["speed"]
                ep_steps[i] += 1

                if done:
                    buf_done[t, i] = 1.0
                    episodes += 1
                    mean_speed = ep_speed_sum[i] / max(ep_steps[i], 1)
                    recent.append(
                        (ep_reward[i], info["progress"], float(info["crashed"]),
                         mean_speed, info["lap_time"], info["lap"],
                         float(info.get("stranded", False)))
                    )
                    if info["lap_time"]:
                        lap_times.append(info["lap_time"])
                    ep_reward[i] = 0.0
                    ep_progress[i] = 0.0
                    ep_speed_sum[i] = 0.0
                    ep_steps[i] = 0.0
                    if info.get("truncated"):
                        pending.append((i, o2))   # Zeitlimit: am Endzustand bootstrap
                    o2 = env.reset()
                else:
                    pending.append((i, o2))
                next_obs[i] = o2

            if pending:
                vals = agent.value(np.stack([o for _, o in pending]))
                for (i, _), v in zip(pending, vals):
                    buf_nextval[t, i] = args.gamma * float(v)

            obs = next_obs
            global_step += N

        # ---------------- PPO-Update ----------------
        advantages, returns = agent.compute_returns(
            buf_rew, buf_val, buf_nextval, buf_done,
            agent.cfg.gamma, agent.cfg.gae_lambda,
        )
        batch = {
            "obs": buf_obs.reshape(T * N, obs_dim),
            "actions": buf_act.reshape(T * N, act_dim),
            "logps": buf_logp.reshape(T * N),
            "advantages": advantages.reshape(T * N),
            "returns": returns.reshape(T * N),
        }
        stats = agent.update(batch)

        # ---------------- Protokoll ----------------
        if global_step - steps_log >= args.log_every or global_step >= args.steps:
            now = time.time()
            fps = (global_step - steps_log) / max(now - t_log, 1e-9)
            steps_log, t_log = global_step, now

            if recent:
                rw = [x[0] for x in recent]
                pr = [x[1] for x in recent]
                cr = [x[2] for x in recent]
                sp = [x[3] for x in recent]
                st = [x[6] for x in recent]
                lt = [x[4] for x in recent if x[4]]
                mean_lt = np.mean(lt) if lt else float("nan")
                line = (
                    f"{global_step:>10} {episodes:>9} {np.mean(rw):>8.1f} "
                    f"{np.mean(pr):>12.1f} {100*np.mean(cr):>7.1f} "
                    f"{100*np.mean(st):>8.1f} "
                    f"{np.mean(sp):>7.1f} {mean_lt:>11.1f} {fps:>6.0f}"
                )
                print(line)
                safe_log(
                    [global_step, episodes, round(np.mean(rw), 3),
                     round(np.mean(pr), 2), round(float(np.mean(cr)), 3),
                     round(float(np.mean(st)), 3),
                     round(float(np.mean(sp)), 3),
                     round(float(mean_lt), 3) if lt else "",
                     agent.updates, round(fps, 1)]
                )

                score = np.mean(rw[-min(len(rw), 20):])
                if score > best_score:
                    best_score = score
                    safe_save({"train_steps": global_step, "score": score},
                              name="best.pt")

        if global_step % args.save_every < N:
            safe_save({"train_steps": global_step})

    # ---------------- Abschluss ----------------
    safe_save({"train_steps": global_step, "score": best_score})
    elapsed = time.time() - t_start
    print(f"\nFertig: {global_step} Schritte, {episodes} Episoden in "
          f"{elapsed:.0f}s ({global_step/elapsed:.0f} FPS)")
    if lap_times:
        print(f"Beste Rundenzeit: {min(lap_times):.1f}s | "
          f"Ø letzte: {np.mean(lap_times[-10:]):.1f}s")
    print(f"Checkpoints: {run_dir/'best.pt'}, {run_dir/'latest.pt'}")
    print(f"Auswerten:   .venv/bin/python -m airacetrack.evaluate "
          f"{run_dir/'best.pt'} --gif lauf.gif")


if __name__ == "__main__":
    main()
