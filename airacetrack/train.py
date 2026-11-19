"""Training starten:

    python -m airacetrack.train --steps 500000

Läuft automatisch mit dem neuesten `best.pt` unter `runs/` weiter (falls
vorhanden), damit nichts neu gelernt werden muss – `--fresh` startet bewusst
bei null, `--resume pfad/…/best.pt` wählt einen bestimmten Checkpoint.

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
                   help="Trainings-Schritte dieses Laufs (zusätzlich zu den "
                        "bereits trainierten)")
    p.add_argument("--num-envs", type=int, default=32, help="Parallele Umgebungen")
    p.add_argument("--rollout", type=int, default=128,
                   help="Schritte pro Umgebung und Rollout")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=None,
                   help="Lernrate (Standard: 3e-4)")
    p.add_argument("--gamma", type=float, default=None,
                   help="Discount-Faktor (Standard: 0.995)")
    p.add_argument("--total-laps", type=int, default=0,
                   help="Episodenende nach N Umläufen (0 = nur Zeitlimit)")
    p.add_argument("--resume", type=str, default=None,
                   help="Checkpoint fortsetzen (.pt) – ohne Angabe wird das "
                        "neueste best.pt unter --out geladen, falls vorhanden")
    p.add_argument("--fresh", action="store_true",
                   help="Nicht fortsetzen, sondern bei null anfangen")
    p.add_argument("--out", type=str, default="runs", help="Ausgabeordner")
    p.add_argument("--log-every", type=int, default=10_000,
                   help="Alle N Schritte Protokoll ausgeben")
    p.add_argument("--save-every", type=int, default=50_000,
                   help="Alle N Schritten Checkpoint speichern")
    return p.parse_args()


def find_latest_checkpoint(out_dir: str | Path) -> Path | None:
    """Neuestes `best.pt` unter `<out>/**` – Fallback: neuestes `latest.pt`."""
    root = Path(out_dir)
    if not root.exists():
        return None

    def mtime(p: Path) -> float:
        return p.stat().st_mtime

    best = sorted(root.glob("**/best.pt"), key=mtime, reverse=True)
    if best:
        return best[0]
    latest = sorted(root.glob("**/latest.pt"), key=mtime, reverse=True)
    return latest[0] if latest else None


def last_log_row(csv_path: Path) -> list[str] | None:
    """Letzte Datenzeile einer CSV – oder None, wenn nicht lesbar/leer."""
    try:
        with csv_path.open(newline="") as f:
            return next(csv.reader(deque(f, maxlen=1)), None)
    except OSError:
        return None


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # Umgebungen
    track = Track(CONTROL_POINTS, **TRACK_CONFIG)
    env_cfg = EnvConfig()
    envs = [RaceEnv(track, env_cfg, seed=args.seed + i) for i in range(args.num_envs)]
    obs_dim, act_dim = envs[0].observation_dim, envs[0].action_dim
    assert (obs_dim, act_dim) == (OBS_DIM, ACT_DIM)

    # Agent – vorhandenen Checkpoint fortsetzen, sonst frisch anlegen
    if args.fresh and args.resume:
        raise SystemExit("--fresh und --resume schließen sich aus.")
    resume_path: Path | None = None
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise SystemExit(f"Checkpoint nicht gefunden: {resume_path}")
    elif not args.fresh:
        resume_path = find_latest_checkpoint(args.out)

    # Zerfall von Lernrate/Entropie über die Schritte dieses Laufs
    updates_per_run = max(args.steps // (args.rollout * args.num_envs), 1)
    payload: dict = {}
    start_step = 0
    episodes = 0
    best_score = -np.inf

    if resume_path is not None:
        agent, payload = PPO.load(str(resume_path))
        if agent.obs_dim != obs_dim or agent.act_dim != act_dim:
            raise SystemExit(
                f"Checkpoint wurde mit {agent.obs_dim} Beobachtungen/"
                f"{agent.act_dim} Aktionen trainiert, die aktuelle Umgebung "
                f"hat {obs_dim}/{act_dim} – Abbruch."
            )
        # Zerfall erneut über diese Laufzeit laufen lassen – sonst stünde die
        # Lernrate am Ende des alten Plans bereits auf ~0 und es passierte
        # beim Fortsetzen nichts mehr.
        agent.cfg.decay_offset = agent.updates
        agent.cfg.lr_decay_steps = agent.updates + updates_per_run
        if args.lr is not None:
            agent.cfg.lr = args.lr
        if args.gamma is not None:
            agent.cfg.gamma = args.gamma
        start_step = int(payload.get("train_steps", 0))
        episodes = int(payload.get("episodes", 0))
        best_score = float(payload.get("score", -np.inf))
        print(f"Fortgesetzt von {resume_path}")
        print(f"  {start_step} Schritte, Update #{agent.updates}, "
              f"Score {best_score:.1f} | Lernrate {agent.cfg.lr:g} zerfällt "
              f"über {updates_per_run} neue Schritte")
    else:
        agent = PPO(obs_dim, act_dim, PPOConfig(
            lr=3e-4 if args.lr is None else args.lr,
            gamma=0.995 if args.gamma is None else args.gamma,
            lr_decay_steps=updates_per_run,
        ))

    # Ausgabeordner – bei einem Resume im Ordner des Checkpoints weiterschreiben
    if resume_path is not None:
        run_dir = resume_path.parent
    else:
        run_dir = Path(args.out) / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.csv"
    # Alte Checkpoint kennen den Episoden-Zähler nicht → aus dem Protokoll
    if resume_path is not None and "episodes" not in payload:
        row = last_log_row(log_path)
        if row and len(row) > 1:
            try:
                episodes = int(float(row[1]))
            except ValueError:
                pass
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
    print(f"Ziel: {start_step} + {args.steps} = {start_step + args.steps} Schritte")
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

    def save_extra() -> dict:
        return {"train_steps": global_step, "episodes": episodes,
                "score": best_score}

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
    global_step = start_step
    target_step = start_step + args.steps
    t_start = time.time()
    t_log = t_start
    steps_log = start_step
    last_save = start_step

    print(f"{'Schritt':>10} {'Episoden':>9} {'Rewards':>8} {'Fortschritt':>12} "
          f"{'Crash%':>7} {'Strand%':>8} {'Tempo':>7} {'Rundenzeit':>11} {'FPS':>6}")

    while global_step < target_step:
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
                    buf_nextval[t, i] = agent.cfg.gamma * float(v)

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
        if global_step - steps_log >= args.log_every or global_step >= target_step:
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

                # Volles 20-Episoden-Fenster nötig: ein Glückstreffer mit
                # zwei kurzen Episoden darf best.pt nicht überschreiben.
                score = np.mean(rw[-min(len(rw), 20):])
                if len(rw) >= 20 and score > best_score:
                    best_score = score
                    safe_save(save_extra(), name="best.pt")

        if global_step - last_save >= args.save_every:
            last_save = global_step
            safe_save(save_extra())

    # ---------------- Abschluss ----------------
    safe_save(save_extra())
    if not (run_dir / "best.pt").exists():   # kurze Läufe: best.pt anlegen
        safe_save(save_extra(), name="best.pt")
    elapsed = time.time() - t_start
    new_steps = global_step - start_step
    print(f"\nFertig: {new_steps} neue Schritte (gesamt {global_step}), "
          f"{episodes} Episoden in {elapsed:.0f}s "
          f"({new_steps/elapsed:.0f} FPS)")
    if lap_times:
        print(f"Beste Rundenzeit: {min(lap_times):.1f}s | "
          f"Ø letzte: {np.mean(lap_times[-10:]):.1f}s")
    print(f"Checkpoints: {run_dir/'best.pt'}, {run_dir/'latest.pt'}")
    print(f"Weitertrainieren: .venv/bin/python -m airacetrack.train --steps 500000")
    print(f"Auswerten:        .venv/bin/python -m airacetrack.evaluate "
          f"{run_dir/'best.pt'} --gif lauf.gif")


if __name__ == "__main__":
    main()
