"""RL-Umgebung: `RaceEnv` im Stil von OpenAI Gym (reset/step).

Beobachtung (18 Werte):
  9 Distanz-Sensoren (Raycast gegen die Streckengrenzen, normalisiert auf 1)
  + normierte Geschwindigkeit
  + sin/cos des Winkelfehlers zur Streckenrichtung
  + lateraler Versatz von der Zentralelinie
  + Krümmung 10/30/60 m voraus (die KI "sieht" die kommende Kurve und
    kann früh genug bremsen)
  + die zuletzt ausgeführten Aktionen (Lenken, Gas)

Belohnung:
  + Fortschritt in Metern (Hauptanreiz, dichtes Signal)
  + kleine Geschwindigkeitsprämie (Tempo)
  - Zeitkosten pro Schritt (Gegenanreiz gegen Stille-Trippeln)
  + großer Bonus pro abgeschlossenem Umlauf
  - Strafe fürs Fahren auf Gras (Minderung + Motorleistung weg)
  - große Strafe bei Unfall (Absturz in die Mauer) → Episodenende
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .car import Car, CarConfig
from .track import Track


@dataclass
class EnvConfig:
    dt: float = 0.05               # Simulations-/Aktionsschritt [s]
    max_steps: int = 1600          # Episodenlänge (80 s)
    n_rays: int = 9                # Sensoren
    ray_max: float = 75.0          # Sensor-Reichweite [m]
    ray_spread_deg: float = 100.0  # Winkel-Ausbreitung ±
    curv_lookaheads: tuple = (10.0, 30.0, 60.0)  # Krümmungs-Vorschau [m]
    runoff: float = 6.0            # Überstand bis zur Mauer (Unfall) [m]
    # Start
    start_speed: float = 30.0      # Mindest-Anfangsgeschwindigkeit [m/s]
    random_start: bool = True      # Startposition zufällig auf der Strecke
    lateral_noise: float = 2.0     # Start-Seitenversatz-Zufall [m]
    speed_noise: float = 8.0       # Startgeschwindigkeit ± [m/s]
    # Strandung: wer dauerhaft stillsteht, beendet die Episode
    stranded_speed: float = 1.0    # Schwelle [m/s]
    stranded_steps: int = 60       # wie viele Schritte unter der Schwelle (3 s)
    # Belohnungsgewichte
    w_progress: float = 0.10
    w_speed: float = 0.005         # Tempobelohnung: zahlt sich sofort aus
    w_time: float = -0.01
    w_lap: float = 30.0
    w_grass: float = -0.4
    w_crash: float = -60.0         # Unfall = teuer (Anreiz sauber zu fahren)
    w_stranded: float = -25.0      # Stillstand = Episode gescheitert
    w_smooth: float = -0.05

    @property
    def ray_angles(self) -> np.ndarray:
        half = np.deg2rad(self.ray_spread_deg)
        return np.linspace(-half, half, self.n_rays)


OBS_DIM = 18
ACT_DIM = 2  # [Lenken, Gas/Bremse] jeweils [-1, 1]


class RaceEnv:
    """Eine Rennstrecke zum Trainieren einer KI."""

    observation_dim = OBS_DIM
    action_dim = ACT_DIM

    def __init__(self, track: Track, config: EnvConfig | None = None,
                 seed: int | None = None) -> None:
        self.track = track
        self.cfg = config or EnvConfig()
        self.car = Car(CarConfig())
        self.rng = np.random.default_rng(seed)

        self._ray_angles = self.cfg.ray_angles
        self._prev_action = np.zeros(2, dtype=np.float32)

        # Episoden-Zustand
        self.steps = 0
        self.total_progress = 0.0
        self.lap = 0
        self.next_cp = 1
        self._prev_s = 0.0
        self._lap_start_time = 0.0
        self._time = 0.0

    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        cfg = self.cfg
        t = self.track
        if cfg.random_start:
            s0 = float(self.rng.uniform(0.0, t.length))
            idx = int(s0 / t.spacing) % t.n
        else:
            idx = 0
        lat = float(self.rng.uniform(-cfg.lateral_noise, cfg.lateral_noise))
        pos = t.center[idx] + t.normal[idx] * lat
        heading = np.arctan2(t.tangent[idx, 1], t.tangent[idx, 0])
        heading += float(self.rng.normal(0.0, 0.03))
        speed = cfg.start_speed + float(self.rng.uniform(-cfg.speed_noise,
                                                          cfg.speed_noise))
        self.car.reset(pos[0], pos[1], heading, max(speed, 10.0))

        self.steps = 0
        self.total_progress = 0.0
        self.lap = 0
        self.next_cp = 1
        self._prev_s = float(t.s[idx])
        self._time = 0.0
        self._lap_start_time = 0.0
        self._stranded_count = 0
        self._prev_action = np.zeros(2, dtype=np.float32)
        return self._observe()

    # ------------------------------------------------------------------
    def step(self, action) -> tuple[np.ndarray, float, bool, dict]:
        cfg = self.cfg
        t = self.track
        action = np.clip(np.asarray(action, dtype=float).reshape(2), -1.0, 1.0)

        idx, lat, s = t.nearest(self.car.position)
        on_grass = abs(lat) > t.half_width
        crashed = abs(lat) > t.half_width + cfg.runoff

        if not crashed:
            self.car.step(action[0], action[1], cfg.dt, on_grass=on_grass)
            self.steps += 1
            self._time += cfg.dt

        # Strandungs-Erkennung: dauerhaftes Stillständen beendet die Episode
        if self.car.speed < cfg.stranded_speed:
            self._stranded_count += 1
        else:
            self._stranded_count = 0

        # Fortschritt
        idx_new, lat_new, s_new = t.nearest(self.car.position)
        delta_s = t.wrap_delta_s(self._prev_s, s_new)
        self.total_progress += delta_s
        self._prev_s = s_new

        # Checkpoints / Umläufe
        lap_time = None
        while (self.next_cp < len(t.checkpoint_s)
               and self.total_progress >= self.lap * t.length + t.checkpoint_s[self.next_cp]):
            self.next_cp += 1
        if self.total_progress >= (self.lap + 1) * t.length:
            self.lap += 1
            self.next_cp = 1
            lap_time = self._time - self._lap_start_time
            self._lap_start_time = self._time

        # Belohnung
        reward = cfg.w_progress * delta_s
        reward += cfg.w_speed * self.car.speed
        reward += cfg.w_time
        reward += cfg.w_smooth * abs(action[0] - self._prev_action[0])
        if on_grass:
            reward += cfg.w_grass
        if self.lap > 0 and lap_time is not None:
            reward += cfg.w_lap

        done = False
        info = {
            "crashed": False,
            "stranded": False,
            "on_grass": on_grass,
            "lap": self.lap,
            "lap_time": lap_time,
            "progress": self.total_progress,
            "speed": self.car.speed,
            "time": self._time,
        }

        if crashed:
            done = True
            reward += cfg.w_crash
            info["crashed"] = True
        elif self._stranded_count >= cfg.stranded_steps:
            done = True
            reward += cfg.w_stranded
            info["stranded"] = True
        elif self.steps >= cfg.max_steps:
            done = True
            info["truncated"] = True

        self._prev_action = action.astype(np.float32)
        obs = self._observe(idx=idx_new, lat=lat_new, s=s_new)
        return obs, float(reward), done, info

    # ------------------------------------------------------------------
    def _observe(self, idx: int | None = None, lat: float | None = None,
                 s: float | None = None) -> np.ndarray:
        t = self.track
        if idx is None:
            idx, lat, s = t.nearest(self.car.position)

        # Sensoren
        angles = self._ray_angles + self.car.heading
        dists = t.raycast(self.car.position, angles, self.cfg.ray_max)
        dists = dists / self.cfg.ray_max

        # Ausrichtung zur Strecke
        track_heading = np.arctan2(t.tangent[idx, 1], t.tangent[idx, 0])
        h_err = self.car.heading - track_heading
        h_err = (h_err + np.pi) % (2.0 * np.pi) - np.pi

        curvatures = [t.curvature_at(idx, la) * 50.0
                      for la in self.cfg.curv_lookaheads]
        v_norm = self.car.speed / self.car.cfg.vmax

        obs = np.concatenate(
            [
                dists,
                [v_norm, np.sin(h_err), np.cos(h_err),
                 lat / t.half_width, *curvatures,
                 self._prev_action[0], self._prev_action[1]],
            ]
        ).astype(np.float32)
        return np.clip(obs, -5.0, 5.0)
