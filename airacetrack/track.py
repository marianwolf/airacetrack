"""Strecken-Modul.

Die Strecke wird als geschlossene Kontrollpunkt-Liste definiiert und mit
einer Catmull-Rom-Spline zur glatten Zentralelinie aufgelöst. Daraus
leiten wir ab:

* Zentralelinie + Fortschritts-Koordinate s (0 .. Streckenlänge)
* linke/rechte Streckengrenze (Begrenzungsmuren)
* Krümmung entlang der Strecke (für Beobachtungen)
* Raycasting gegen die Begrenzungen (Lidar-ähnliche Distanz-Sensoren)
* Checkpoints für die gültige Umlauf-Zählung
"""

from __future__ import annotations

import numpy as np


def _catmull_rom_closed(points: np.ndarray, samples_per_seg: int = 24) -> np.ndarray:
    """Geschlossene Catmull-Rom-Spline durch die Kontrollpunkte."""
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    out = np.empty((n * samples_per_seg, 2), dtype=float)
    k = 0
    for i in range(n):
        p0 = pts[(i - 1) % n]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i + 2) % n]
        for j in range(samples_per_seg):
            t = j / samples_per_seg
            t2 = t * t
            t3 = t2 * t
            out[k] = 0.5 * (
                2.0 * p1
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
            )
            k += 1
    return out


def _resample_closed(pts: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Geschlossene Polyline gleichmäßig entlang der Bogenlänge neu abtasten."""
    closed = np.vstack([pts, pts[:1]])
    seg = np.diff(closed, axis=0)
    d = np.linalg.norm(seg, axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    total = float(s[-1])
    n = max(int(round(total / spacing)), 32)
    target = np.linspace(0.0, total, n, endpoint=False)
    x = np.interp(target, s, closed[:, 0])
    y = np.interp(target, s, closed[:, 1])
    return np.stack([x, y], axis=1), target


def _box_smooth(a: np.ndarray, window: int) -> np.ndarray:
    """Kastenglättung auf einem geschlossenen Array."""
    kernel = np.ones(window) / window
    tiled = np.tile(a, 3)
    smoothed = np.convolve(tiled, kernel, mode="same")
    n = len(a)
    return smoothed[n : 2 * n]


class Track:
    """Eine Rennstrecke mit Begrenzungen, Checkpoints und Sensoren."""

    def __init__(
        self,
        control_points,
        width: float = 14.0,
        spacing: float = 2.5,
        checkpoint_spacing: float = 120.0,
    ) -> None:
        self.width = float(width)
        self.half_width = self.width / 2.0
        self.spacing = float(spacing)

        raw = _catmull_rom_closed(np.asarray(control_points, dtype=float), 24)
        self.center, self.s = _resample_closed(raw, self.spacing)
        self.n = len(self.center)
        self.length = float(self.n * self.spacing)

        # Tangenten / Normalen (Normale zeigt nach links)
        nxt = np.roll(self.center, -1, axis=0)
        tang = nxt - self.center
        tang /= np.linalg.norm(tang, axis=1, keepdims=True)
        self.tangent = tang
        self.normal = np.stack([-tang[:, 1], tang[:, 0]], axis=1)

        # Begrenzungen
        self.left = self.center + self.normal * self.half_width
        self.right = self.center - self.normal * self.half_width
        self._seg_a = np.vstack([self.left, self.right])
        self._seg_b = np.vstack(
            [np.roll(self.left, -1, axis=0), np.roll(self.right, -1, axis=0)]
        )
        self._seg_vec = self._seg_b - self._seg_a

        # Krümmung (Vorzeichen: + = nach links), geglättet
        heading = np.arctan2(self.tangent[:, 1], self.tangent[:, 0])
        dh = np.diff(np.concatenate([heading, heading[:1]]))
        dh = (dh + np.pi) % (2.0 * np.pi) - np.pi
        curv = dh / self.spacing
        self.curvature = _box_smooth(curv, 9)

        # Checkpoints: Index 0 = Start/Ziellinie
        cp_every = max(int(round(checkpoint_spacing / self.spacing)), 4)
        self.checkpoint_idx = np.arange(0, self.n, cp_every)
        self.checkpoint_s = self.s[self.checkpoint_idx]

        # Startpose (Start/Ziellinie)
        self.start_pos = self.center[0].copy()
        self.start_heading = float(np.arctan2(self.tangent[0, 1], self.tangent[0, 0]))

    # ------------------------------------------------------------------
    def nearest(self, xy: np.ndarray) -> tuple[int, float, float]:
        """Nächster Punkt auf der Zentralelinie.

        Rückgabe: (Index, lateraler Versatz in m [+ = links], Fortschritt s)
        """
        xy = np.asarray(xy, dtype=float)
        a = self.center
        b = np.roll(a, -1, axis=0)
        ab = b - a
        ap = xy - a
        denom = np.einsum("ij,ij->i", ab, ab)
        t = np.clip(np.einsum("ij,ij->i", ap, ab) / denom, 0.0, 1.0)
        proj = a + t[:, None] * ab
        d = np.linalg.norm(xy - proj, axis=1)
        i = int(np.argmin(d))
        lat = float(np.dot(xy - proj[i], self.normal[i]))
        s = float(self.s[i] + t[i] * self.spacing)
        return i, lat, s

    def curvature_at(self, idx: int, lookahead: float = 25.0) -> float:
        """Krümmung ein Stück vor dem gegebenen Index (gerundet)."""
        off = int(round(lookahead / self.spacing))
        return float(self.curvature[(idx + off) % self.n])

    def raycast(self, xy: np.ndarray, angles: np.ndarray, max_dist: float) -> np.ndarray:
        """Distanz von `xy` in Richtung `angles` bis zur Streckengrenze."""
        xy = np.asarray(xy, dtype=float)
        angles = np.asarray(angles, dtype=float)
        r = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (R, 2)
        s_vec = self._seg_vec[None, :, :]                        # (1, N, 2)
        qp = (self._seg_a - xy)[None, :, :]                      # (1, N, 2)

        rs = r[:, 0][:, None] * s_vec[..., 1] - r[:, 1][:, None] * s_vec[..., 0]
        qp_s = qp[..., 0] * s_vec[..., 1] - qp[..., 1] * s_vec[..., 0]
        qp_r = qp[..., 0] * r[:, 1][:, None] - qp[..., 1] * r[:, 0][:, None]

        with np.errstate(divide="ignore", invalid="ignore"):
            t = qp_s / rs
            u = qp_r / rs
        valid = (rs != 0.0) & (t >= 0.0) & (u >= 0.0) & (u <= 1.0)
        t = np.where(valid, t, np.inf)
        dist = np.min(t, axis=1)
        return np.minimum(dist, max_dist)

    def wrap_delta_s(self, s_old: float, s_new: float) -> float:
        """Signierter Fortschritt über die Startlinie hinweg."""
        d = s_new - s_old
        half = self.length / 2.0
        if d > half:
            d -= self.length
        elif d < -half:
            d += self.length
        return d

    # ------------------------------------------------------------------
    def plot(self, ax, show_centerline: bool = True, show_checkpoints: bool = False):
        """Strecke in eine Matplotlib-Achse einzeichnen."""
        for boundary in (self.left, self.right):
            closed = np.vstack([boundary, boundary[:1]])
            ax.plot(closed[:, 0], closed[:, 1], color="black", lw=2.0, zorder=3)
        if show_centerline:
            closed = np.vstack([self.center, self.center[:1]])
            ax.plot(closed[:, 0], closed[:, 1], color="gray", lw=0.8, ls="--", zorder=2)
        if show_checkpoints:
            ax.scatter(
                self.center[self.checkpoint_idx, 0],
                self.center[self.checkpoint_idx, 1],
                c="red", s=12, zorder=4,
            )
        # Start/Ziellinie
        p = self.center[0]
        n = self.normal[0] * self.half_width
        ax.plot([p[0] - n[0], p[0] + n[0]], [p[1] - n[1], p[1] + n[1]],
                color="green", lw=3, zorder=5)
        return ax
