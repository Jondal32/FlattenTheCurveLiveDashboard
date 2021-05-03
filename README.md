# Dashboard zum Projektseminar "Flatten the Curve"

- [Smart Social Distancing](#smart-social-distancing)
  - [Einleitung](#Einleitung)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
  - [Probleme, Bugs und weitere Aufgaben](#Probleme,-Bugs-und-weitere-Aufgaben)



## Einleitung

Das Live Dashboard Flatten the Curve ist das Ergebnis eines Uni-Projektseminar mit dem Ziel den Einzelhandel bei der Bekämpfung gegen die Ausbreitung von Covid-19 zu unterstützen.
Zum aktuellen Zeitpunkt muss zusätzliches Personal für eine manuelle Einlasskontrolle, Überwachung der Maskenpflicht oder auch zur Einhaltung der Abstandsregeln eingesetzt werden. Dieser Zustand soll durch den Einsatz einer Edge-Intelligence-Lösung auf dem Jetson Nano automatisiert werden.
Da alle Berechnungen auf einem Gerät ausgeführt werden sind so nur minimale Schritte für das Setup nötig und Datenschutz- und Sicherheitsbedenken werden minimiert. Wir haben uns bei der Implementierung für Flask Application entschieden mit einer MySql Datenbank.


## Installation

Für die Installation befinden sich alle relevanten Anforderungen unter [requirements.txt](https://github.com/Jondal32/einfaches_dashboard_feb_2021_2/blob/master/requirements.txt).
Des weiteren wird eine MySQL Datenbank benötigt, welche innerhalb von app.py eingebunden werden muss. Dafür ist die Struktur aus [flask_dashboard.sql](https://github.com/Jondal32/einfaches_dashboard_feb_2021_2/blob/master/static/database/flask_dashboard.sql) zu übernehmen.
Die Datenbank verfügt über ein User und Rollensystem, wodurch bei einer neuen Registrierung der Datenbank Administrator zunächst die entsprechende Rolle in Form von einem Bit-Feld innerhalb der User-Tabelle verändern muss. Der Benutzer wird für die entsprechenden Bereiche innerhalb des Dashboard mit dem Bit-Wert 1 freigeschaltet.
(Der Benutzer "Jondal" mit Passwort "123" ist bereits in der Datenbank hinterlegt und kann alle Funktion ausführen).


## Probleme, Bugs und weitere Aufgaben

* 



### Running the app

```bash
python app.py
```

