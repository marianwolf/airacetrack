"""Strecken-Vorschau rendern:  python -m airacetrack.preview_track"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .track import Track
from .tracks import CONTROL_POINTS, TRACK_CONFIG


def main() -> None:
    parser = argparse.ArgumentParser(description="Strecken-Vorschau rendern")
    parser.add_argument("--out", default="track_preview.png", help="Zieldatei (PNG)")
    args = parser.parse_args()

    track = Track(CONTROL_POINTS, **TRACK_CONFIG)

    fig, ax = plt.subplots(figsize=(11, 9))
    track.plot(ax, show_centerline=True, show_checkpoints=True)
    ax.set_aspect("equal")
    ax.set_title(
        f"airacetrack – {track.length:.0f} m Streckenlänge, "
        f"{track.width:.0f} m Breite, {len(track.checkpoint_idx)} Checkpoints"
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"Gespeichert: {args.out}")
    print(f"Länge: {track.length:.1f} m | Kurzester Radius ≈ "
          f"{1.0 / max(abs(track.curvature).max(), 1e-9):.1f} m")


if __name__ == "__main__":
    main()
