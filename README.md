# Dashboard zum Projektseminar "Flatten the Curve"

- [Dashboard zum Projektseminar "Flatten the Curve"]
  - [Einleitung](#einleitung)
  - [Installation](#installation)
    - [App starten](#app-starten)
  - [Probleme, Bugs und weitere Aufgaben](#probleme-und-bugs)



## Einleitung

Das Live Dashboard Flatten the Curve ist das Ergebnis eines Uni-Projektseminar mit dem Ziel den Einzelhandel bei der Bekämpfung gegen die Ausbreitung von Covid-19 zu unterstützen.
Zum aktuellen Zeitpunkt muss zusätzliches Personal für eine manuelle Einlasskontrolle, Überwachung der Maskenpflicht oder auch zur Einhaltung der Abstandsregeln eingesetzt werden. Dieser Zustand soll durch den Einsatz einer Edge-Intelligence-Lösung auf dem Jetson Nano automatisiert werden.
Da alle Berechnungen auf einem Gerät ausgeführt werden sind so nur minimale Schritte für das Setup nötig und Datenschutz- und Sicherheitsbedenken werden minimiert. Wir haben uns bei der Implementierung für Flask Application mit einer MySql Datenbank entschieden. Die Automatisierung der Aufgaben umfasst dabei die automatisierte Erkennung von Masken, einen Person Zähler und die Kontrolle von Abständen zwsichen Personen.


## Installation

Für die Installation befinden sich alle relevanten Anforderungen unter [requirements.txt](https://github.com/Jondal32/einfaches_dashboard_feb_2021_2/blob/master/requirements.txt).
Des weiteren wird eine MySQL Datenbank benötigt, welche innerhalb von app.py eingebunden werden muss. Dafür ist die Struktur aus [flask_dashboard.sql](https://github.com/Jondal32/einfaches_dashboard_feb_2021_2/blob/master/static/database/flask_dashboard.sql) zu übernehmen.
Die Datenbank verfügt über ein User und Rollensystem, wodurch bei einer neuen Registrierung der Datenbank Administrator zunächst die entsprechende Rolle in Form von einem Bit-Feld innerhalb der User-Tabelle verändern muss. Der Benutzer wird für die entsprechenden Bereiche innerhalb des Dashboard mit dem Bit-Wert 1 freigeschaltet.
(Der Benutzer "Jondal" mit Passwort "123" ist bereits in der Datenbank hinterlegt und kann alle Funktion ausführen).
Bei einer Anwendung über den Jetson Nano muss zusätzlich noch Jetson-Inference installiert werden und die Pfade in app.py entsprechend angepasst, sodass aus core anstatt der regulären Python Skripte die für den Jetson abgestimmten übernommen werden.
Vor der Anwendung muss außerdem ein [models](https://drive.google.com/drive/folders/1bvcnWpXHDkKYeLWoLm6CNbMQfvspHzQi?usp=sharing) Ordner in das Repository eingefügt werden. Darin befinden sich bereits trainierte Machine-Learning Modelle für Maskenerkennung als auch Personen Zähler und die Abstandskontrolle.

### App starten

Nachdem die Datenbank installiert, aktiviert und die vorgegebene Datenstruktur migriert wurde kann die App gestartet werden.
```bash
python app.py
```

## Probleme und Bugs

#### Maskenerkennung

* Je weiter man sich von der Kamera entfernt, desto unsicherer wird die Vorhersage beziehungsweise generell ein Gesicht zu erkennen.
* Personen mit Maske können trotz Tracking nur schwer zugeordnet werden, wodurch eine genau Anzahl von Personen mit und ohne Maske fast unmöglich ist. Daher zunächst nur eine Verteilung des Verhältnis zwischen Personen mit und ohne Maske über die verarbeitenen FPS hinweg.
* Genauigkeit mit SSD-MobileNet V2 300x300 ausbaufähig, jedoch bisher der schnellste getestete Ansatz (als TensorRT Modell auf dem Jetson ebenfalls am schnellsten)


#### Personen Zähler
* Vogelperspektive mit leicht schrägen Winkel erlaubt beste Vorhersage beziehungsweise Genauigkeit der Erkennung von Personen
* das auf dem COCO Datensatz trainierte SSD-MobileNet bei der Erkennung der Personen oft ungenau beziehungsweise erkennt diese zu spät oder garnicht
* Performance ausbaufähig
* Tracking mit Dlib auf dem Jetson Nano sehr langsam und das Bottle Neck (Detection kann auf Flask flüssig mit etwa 15-18 FPS laufen; das gesamte Skripte jedoch nur mit 5-6 FPS, da das Tracking über die CPU berechnet wird)

#### Abstandskontrolle

* MobileNet am schnellsten, hat jedoch Probleme Personen in weiter Entfernung zu identifizieren und konstant zu erkennen.
* Yolo V3/V4 hat zwar von den getesteten Modellen die mit Abstand größte Genauigkeit (die Tiny Version ist deutlich schlechter), jedoch auf dem Jetson Nano selbst ohne zusätzliche Flask Applikation sehr langsam (Yolo v3: ~1.8-2 FPS, Yolo v3 Tiny ~10-15 FPS, Yolo v4: ~1 FPS)
* Kamerawinkel und entsprechender Abstand der Bounding-Boxes entscheidend für die Genauigkeit der Verstöße; erkennt keine Paare oder zusammengehörige Personen -> als geeignetes Mittel fraglich, kann lediglich Informationen über potentielle Bereiche mit gehäuften Abstandsunterschreitungen geben

#### Generelle Fehler

* Aufgrund der Umsetzung mit Flask und OpenCV wird viel Speicher
* Wenn man in Flask die Raspberry Pi Kamera mittels GStream ansteuert (einzige Möglichkeit) bekommt man zwar erstmalig Bild, jedoch lässt sich der GStream durch ein einfaches VideoCapture.release() nicht mehr schließen und die gesamte App stürzt ab. Führt man ein Python Script mit gleicher Logik einzeln aus, so tritt dieses Problem nicht auf, da beim Beenden auch sofort der GStream geschlossen wird.
* GStreamer lässt keine parallelen Streams zu beziehungsweise nacheinander aufgerufene Streams
* Aufgrund von Hintergrundaktivitäten durch das Dashboard bekommt man deutlich weniger FPS bei der Objekterkennung als wenn diese einzeln in einem Fenster ausgeführt werden.
* Sobald der Personen Zähler erstmalig gelaufen ist belegt das Script ungewollt auch weiterhin Rechenleistung und Speicherplatz, wodurch die anderen Funktionen nur noch mit erheblichen niedrigen FPS funktionieren - nach einem Neustart/Reload funktioniert es wieder wie zuvor.





