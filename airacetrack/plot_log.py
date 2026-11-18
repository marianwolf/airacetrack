"""Lernkurven aus dem Trainings-Protokoll plotten:

    python -m airacetrack.plot_log runs/<timestamp> [--out lernkurve.png]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description="Trainings-Protokoll plotten")
    p.add_argument("run_dir", help="Ordner mit log.csv (oder direkter CSV-Pfad)")
    p.add_argument("--out", default=None, help="Zieldatei (PNG)")
    args = p.parse_args()

    csv_path = Path(args.run_dir)
    if csv_path.is_dir():
        csv_path = csv_path / "log.csv"
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None,
                         encoding="utf-8")
    out = args.out or str(csv_path.with_name("lernkurve.png"))

    step = data["global_step"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    ax = axes[0, 0]
    ax.plot(step, data["mean_reward"], color="#e10600")
    ax.set_title("Ø Episoden-Reward")
    ax = axes[0, 1]
    ax.plot(step, data["mean_progress"], color="#00a3e0")
    ax.set_title("Ø Fortschritt [m pro Episode]")
    ax = axes[1, 0]
    ax.plot(step, 100 * data["crash_rate"], label="Unfall", color="black")
    ax.plot(step, 100 * data["stranded_rate"], label="Strandung", color="gray")
    ax.set_title("Episode-Enden [%]")
    ax.legend()
    ax = axes[1, 1]
    ax.plot(step, 3.6 * data["mean_speed"], label="Ø Tempo", color="#00a3e0")
    valid = np.isfinite(data["mean_lap_time"]) & (data["mean_lap_time"] > 0)
    if valid.any():
        ax2 = ax.twinx()
        ax2.plot(step[valid], data["mean_lap_time"][valid], color="#e10600",
                 label="Ø Rundenzeit")
        ax2.set_ylabel("Rundenzeit [s]", color="#e10600")
    ax.set_title("Tempo [km/h] / Rundenzeit")
    ax.set_ylabel("km/h")

    for a in axes.flat:
        a.grid(alpha=0.3)
        a.set_xlabel("Trainingsschritte")
    fig.suptitle(f"airacetrack – {csv_path.parent.name}")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"Gespeichert: {out}")


if __name__ == "__main__":
    main()
