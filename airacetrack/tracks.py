"""Formel-1-inspirierter Streckenentwurf (geschlossener Circuit, gegen den Uhrzeigersinn).

Alle Punkte in Metern. Die Reihenfolge bildet die Fahrtrichtung ab:

* Lange Start-/Zielgerade am Süden (Richtung Osten)
* T1 – schnelle Linkskurve nach Nordosten
* Ostseite mit Serpentine (schnelle Richtungswechsel)
* Nordost-Ecke nach Westen
* Haarnadel (180°-Rechtskurve, Radius ≈ 45 m) als Einstich in die Infield-Sektion
* Weiter nach Westen, Südseite der Infield-Ausfahrt
* Westseite nach Süden
* Südwest-Kehre zurück auf den Startgerader
"""

CONTROL_POINTS = [
    # Start-/Zielgerader (Richtung Ost)
    (140.0, 45.0),
    (330.0, 60.0),
    # T1: schnelle Linkskurve
    (470.0, 95.0),
    (565.0, 175.0),
    # Ostseite mit Serpentine (schnelle Richtungswechsel)
    (610.0, 285.0),
    (575.0, 350.0),
    (605.0, 420.0),
    # Nordost-Ecke, nun Richtung Westen
    (555.0, 500.0),
    (515.0, 535.0),
    (478.0, 553.0),
    # Einfahrt in die Haarnadel (rundet weich nach Süden ab)
    (447.0, 547.0),
    (434.0, 515.0),
    (430.0, 470.0),
    (430.0, 400.0),
    # Haarnadel: 180°-Rechtskurve, Radius ≈ 45 m
    (420.0, 340.0),
    (385.0, 305.0),
    (345.0, 335.0),
    # Ausfahrt nach Norden
    (340.0, 400.0),
    (342.0, 480.0),
    (345.0, 552.0),
    # Weiter nach Westen
    (270.0, 570.0),
    (170.0, 560.0),
    (90.0, 525.0),
    # Westseite nach Süden
    (45.0, 450.0),
    (30.0, 350.0),
    (28.0, 250.0),
    (31.0, 165.0),
    # Südwest-Kehre (Radius ≈ 55 m) zurück auf den Startgerader
    (33.0, 110.0),
    (49.0, 71.0),
    (88.0, 55.0),
]

# Eigenschaften der Strecke (an Track() übergebene Argumente)
TRACK_CONFIG = {
    "width": 14.0,          # Fahrbahnbreite in Metern
    "spacing": 2.5,         # Abtastung der Zentralelinie in Metern
    "checkpoint_spacing": 120.0,
}
