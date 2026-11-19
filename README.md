# airacetrack 🏎️

**Eine KI, die man auf einer Formel-1-Strecke trainiert** – ein kompletter
2D-Rennsimulator mit Reinforcement Learning (PPO) in Python + PyTorch.

Die KI steuert ein Formel-1-artiges Fahrzeug (kinematisches Fahrradmodell mit
Downforce-Limit, Bremsen, Luftwiderstand und Gras-Effekt) und lernt **von
null selbst**, die Strecke sauber und schnell zu fahren – inklusive Bremspunkten,
Kurvenlinien und Umlaufzeit.

```
Strecke: 2328 m, Haarnadel, Serpentine, langer Start-/Zielgerader
Beobachtung: 9 Lidar-Sensoren + Geschwindigkeit + Streckenlage + Krümmungsvorschau
Aktion:      [Lenken, Gas/Bremse]
Algorithmus: PPO (Actor-Critic, tanh-gedämpfte Aktionen, GAE)
```

## Schnellstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Strecke ansehen
python3 -m airacetrack.preview_track --out track.png

# KI trainieren (Standard: 500k Schritte, dauert auf dem CPU ~5 Minuten)
python3 -m airacetrack.train --steps 500000
```

Ohne weitere Angaben lädt `train` **das neueste `best.pt` unter `runs/`** und
fährt dort weiter – nichts wird neu gelernt, `--steps` sind dann die *zusätzlichen*
Schritte dieses Laufs. Wer bewusst neu anfangen will: `--fresh`.

```bash
python3 -m airacetrack.train --steps 500000     # fortsetzen (auto)
python3 -m airacetrack.train --fresh --steps 500000   # bei null anfangen
python3 -m airacetrack.train --resume runs/20261118-163645/best.pt --steps 100000

# Ergebnis auswerten
python3 -m airacetrack.evaluate runs/<timestamp>/best.pt
python3 -m airacetrack.evaluate runs/<timestamp>/best.pt --gif lauf.gif
python3 -m airacetrack.evaluate runs/<timestamp>/best.pt --plot traj.png
python3 -m airacetrack.evaluate runs/<timestamp>/best.pt --live
```

## Wie es funktioniert

### Die Strecke (`airacetrack/track.py`, `tracks.py`)

Die Strecke wird als Liste von Kontrollpunkten definiert und mit einer
Catmull-Rom-Spline zur Zentralelinie aufgelöst (Abtastung 2,5 m). Daraus
entstehen automatisch:

* die **Begrenzungsmuren** (linke/rechte Fahrbahnkante, Breite 14 m),
* die **Fortschritts-Koordinate** `s` (für Belohnung und Umlaufzählung),
* die **Krümmung** entlang der Strecke (als „Fernsicht" für die KI),
* **Checkpoints** (120 m), die für einen gültigen Umlauf passiert werden müssen,
* **Raycasting** gegen die Muren – die Distanz-Sensoren der KI.

Eine eigene Strecke? Kontrollpunkte in `airacetrack/tracks.py` ändern und mit
`preview_track` prüfen – mehr Punkte = engere Kurven, ungleichmäßige Abstände =
geradere Passagen.

### Das Fahrzeug (`airacetrack/car.py`)

* Antrieb bis 14 m/s², Bremsen bis 26 m/s², Luftwiderstand `a = 0,0018·v²`
* **Downforce-Limit**: der Lenkwinkel wird je nach Tempo geklemmt, damit die
  Seitenkraft 26 m/s² nicht überschreitet → wer zu schnell in die Kurve kommt,
  untersteuert und fliegt raus
* Lenkung folgt dem Kommando mit endlicher Rate (3 rad/s)
* **Gras** kostet Motorleistung und Speed – wer die Strecke verlässt, bremst ab

### Die Umgebung (`airacetrack/env.py`)

Gym-kompatible API: `reset() → obs`, `step(action) → obs, reward, done, info`.

**Beobachtung (18 Werte)**

| Werte | Bedeutung |
|---|---|
| 9 | Distanz-Sensoren ±100°, Reichweite 75 m |
| 1 | Geschwindigkeit (normiert) |
| 2 | sin/cos des Winkelfehlers zur Streckenrichtung |
| 1 | lateraler Versatz von der Mittellinie |
| 3 | Krümmung 10/30/60 m voraus (sieht die Kurve „vorher") |
| 2 | zuletzt ausgeführte Aktionen |

**Belohnung pro Schritt** (0,05 s)

```
+0,10 · Fortschritt in Metern      (Hauptanreiz)
+0,004 · Geschwindigkeit           (Tempo zahlt sich sofort aus)
−0,018                            (Zeitkosten gegen Stillstand → senkt die Ø Zeit)
+30 + 2,5 · (60 s − Rundenzeit)   (Bonus pro Umlauf, schnellere Runde = mehr;
                                    Untergrenze 0, damit Fahren immer lohnt)
−0,40 · jede Sekunde auf Gras       (Motorleistung weg)
−60                            (Unfall → Episodenende)
−70                            (Strandung: >3 s Stillstand = gescheitert)
```

Die Gewichte sind bewusst Richtung **Rundenzeit und Ø-Zeit** verschoben:
Rundenzeit-Bonus und Zeitkosten sind stärker, die reine Momentan-Speed-Prämie
etwas reduziert – Anreiz ist die tatsächliche Umlaufzeit, nicht das Tempo
auf der Geraden.

Episoden starten **zufällig auf der ganzen Strecke** (20–40 m/s), damit die KI
Kurven aus allen Situationen heraus lernen muss und nicht nur das Ausrollen vom
Start.

### Das Training (`airacetrack/ppo.py`, `policy.py`, `train.py`)

* **PPO** mit GAE(λ), Clipping 0,2, 4 Epochen, Minibatches
* **Actor-Critic-MLP** 128–128, tanh-aktiviert, Orthogonal-Init
* **Aktionen** sind Gauss-Verteilungen, per `tanh` auf [-1, 1] begrenzt
  (korrekte Log-Wahrscheinlichkeits-Dämpfung)
* **Beobachtungs-Normalisierung** mit laufendem Mittelwert/Varianz – die
  normalisierten Werte werden im Buffer gespeichert, damit die
  PPO-Verhältnisse exakt passen
* 16 parallele Umgebungen, 128 Schritte je Rollout (2048 Samples je Update)
* Linearer Lernraten-Zerfall über die Laufzeit, Entropie-Bonus 0,02
* Checkpoints: `runs/<timestamp>/latest.pt` (fortsetzbar) und `best.pt` –
  `train` startet automatisch mit dem neuesten `best.pt` weiter
  (Schritt-Zähler, Score-Grenze und Lernraten-Plan laufen sauber weiter)

## CLI-Optionen

```bash
# Trainieren
python -m airacetrack.train \
    --steps 2000000 \          # Schritte dieses Laufs (zusätzlich zu den
                               # bereits trainierten – automatisch fortgesetzt)
    --num-envs 32 \            # parallele Umgebungen
    --rollout 256 \            # Schritte je Rollout
    --gamma 0.99 \             # Discount-Faktor
    --fresh \                  # nicht fortsetzen, sondern neu anfangen
    --resume runs/.../best.pt  # bestimmten Checkpoint laden

# Auswerten
python -m airacetrack.evaluate pfad/best.pt \
    --episodes 5 --laps 3 \    # 5 Episoden à 3 Umläufe
    --stochastic \             # Aktionen samplen statt deterministisch
    --live                     # Live ansehen (interaktives Fenster)
```

Nach dem Training wird die Auswertung mit `best.pt` direkt als GIF, PNG
(Fahrtrajektorie, eingefärbt nach Tempo) oder Live-Fenster angeboten.
`log.csv` im Ausgabeordner enthält Reward, Fortschritt, Crash-Quote, Tempo und
Rundenzeiten – direkt plotten mit:

```bash
.venv/bin/python -m airacetrack.plot_log runs/<timestamp>
```

## Projektstruktur

```
airacetrack/
├── tracks.py         # Strecken-Layout (Kontrollpunkte) – hier Strecke ändern
├── track.py          # Spline, Muren, Checkpoints, Raycasting, Krümmung
├── car.py            # Fahrzeugphysik (kinematisches Fahrradmodell)
├── env.py            # RL-Umgebung: Beobachtung, Belohnung, Episodenlogik
├── policy.py         # Actor-Critic-Netz (PyTorch)
├── ppo.py            # PPO-Algorithmus, Normalisierung, Checkpoints
├── train.py          # Trainingsschleife (CLI)
├── evaluate.py       # Modell auswerten: GIF, Plot, Live
├── plot_log.py       # Lernkurven aus log.csv plotten
└── preview_track.py  # Strecke als PNG rendern
```

## Trainings-Tipps

* **Kurztest**: `--steps 50000` (~30 s) – zeigt schon Fortschritt
  (Crash-Quote und Tempo steigen).
* **Gute Ergebnisse**: 1–2 Mio. Schritte. Die KI fährt dann saubere
  Runden und sucht aktiv den Rhythmus.
* **Schwieriger machen**: `EnvConfig` in `train.py` (z. B. `w_grass` senken,
  `runoff` verkleinern, `width` auf 10 m).
* **Strecke härter**: mehr engere Kontrollpunkte in `tracks.py`.
* **Modell weitertrainieren**: einfach nochmal `python -m airacetrack.train
  --steps 500000` starten – das neueste `best.pt` wird automatisch geladen
  (`--resume pfad/…/best.pt` wählt einen anderen, `--fresh` beginnt neu).
  Am Ende des Laufs stehen wieder `best.pt` und `latest.pt` im selben Ordner.
