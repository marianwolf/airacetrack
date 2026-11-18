"""Fahrzeug-Physik: kinematisches Fahrradmodell im Stil eines Formel-1-Wagens.

Vereinfachungen gegenüber der Realität, aber mit den relevanten Effekten:

* Antrieb, Bremsen, Luftwiderstand
* Lenkwinkel wird begrenzt durch die verfügbare Seitenkraft (Downforce!)
* Lenkradwinkel-Änderung ist ratenbegrenzt (Lenkung braucht Zeit)
* Untersteuern bei zu hoher Kurvengeschwindigkeit entsteht automatisch,
  weil der Winkel an die Maximal-Seitenbeschleunigung gekoppelt ist
* Gras/Rasen kostet starken Antrieb (langsamer, rutschig)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CarConfig:
    wheelbase: float = 2.7        # Radstand [m]
    width: float = 1.9            # Fahrzeugbreite [m]
    max_steer: float = 0.55       # maximaler Lenkwinkel [rad]
    steer_rate: float = 3.0       # Lenkwinkel-Änderung [rad/s]
    max_accel: float = 14.0       # Antriebsbeschleunigung [m/s²]
    max_brake: float = 26.0       # Bremsbeschleunigung [m/s²]
    drag_c: float = 0.0018        # Luftwiderstand: a = drag_c * v²
    rolling: float = 0.5          # Rollwiderstand [m/s²]
    a_lat_max: float = 26.0       # Max. Seitenbeschleunigung (Downforce) [m/s²]
    vmax: float = 95.0            # harte Geschwindigkeitsbegrenzung [m/s]
    grass_extra_drag: float = 9.0 # Zusatzverzögerung auf Gras [m/s²]


class Car:
    """Ein einzelnes Fahrzeug mit kinematischem Fahrradmodell."""

    def __init__(self, config: CarConfig | None = None) -> None:
        self.cfg = config or CarConfig()
        # Zustand
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0
        self.speed = 0.0
        self.steer = 0.0

    # ------------------------------------------------------------------
    def reset(self, x: float, y: float, heading: float, speed: float = 0.0) -> None:
        self.x = float(x)
        self.y = float(y)
        self.heading = float(heading)
        self.speed = float(speed)
        self.steer = 0.0

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)

    @property
    def velocity(self) -> np.ndarray:
        return self.speed * np.array([np.cos(self.heading), np.sin(self.heading)])

    # ------------------------------------------------------------------
    def step(self, steer_cmd: float, throttle_cmd: float, dt: float,
             on_grass: bool = False) -> dict:
        """Einen Zeitschritt ausführen.

        steer_cmd, throttle_cmd: Werte in [-1, 1]
        """
        c = self.cfg
        steer_cmd = float(np.clip(steer_cmd, -1.0, 1.0))
        throttle_cmd = float(np.clip(throttle_cmd, -1.0, 1.0))

        # Ziel-Lenkwinkel...
        target = c.max_steer * steer_cmd
        # ...begrenzt durch die zulässige Seitenkraft (v²/r = g-Leistung):
        # a_lat = v² * tan(δ) / L  =>  δ_max(v) = atan(a_lat_max * L / v²)
        grip_limit = np.arctan(c.a_lat_max * c.wheelbase / max(self.speed**2, 1.0))
        target = float(np.clip(target, -grip_limit, grip_limit))
        # Lenkrad folgt dem Ziel mit endlicher Rate
        max_delta = c.steer_rate * dt
        self.steer += float(np.clip(target - self.steer, -max_delta, max_delta))

        # Längsdynamik
        if throttle_cmd >= 0.0:
            accel = throttle_cmd * c.max_accel
        else:
            accel = throttle_cmd * c.max_brake  # negativ = bremsen
        accel -= c.drag_c * self.speed**2
        if self.speed > 0.0:
            accel -= c.rolling
        if on_grass:
            accel -= c.grass_extra_drag
            accel = min(accel, 2.0)  # Antrieb auf Gras stark begrenzt

        self.speed = float(np.clip(self.speed + accel * dt, 0.0, c.vmax))

        # Kinematik
        yaw_rate = self.speed * np.tan(self.steer) / c.wheelbase
        self.heading += yaw_rate * dt
        # Heading in [-pi, pi] halten
        self.heading = (self.heading + np.pi) % (2.0 * np.pi) - np.pi
        self.x += self.speed * np.cos(self.heading) * dt
        self.y += self.speed * np.sin(self.heading) * dt

        return {
            "steer": self.steer,
            "throttle": throttle_cmd,
            "yaw_rate": yaw_rate,
        }
