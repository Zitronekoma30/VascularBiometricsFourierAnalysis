# Gruppe Fourier Transform
*Andrassik, Krall*

![comparison](https://github.com/user-attachments/assets/69e38426-d94d-450b-9b4f-3c5657238fd3)

### Notizen:
Grundlagen Bildverarbeitung - Besprechungen

Abschlusspräsentation:
- nur ein Band rekonstruieren

------------------------------------------------------------------------------------------------------

30.10.\
Bandpassfilter -> Ring -> Koeffizient -> Mittelwert/Abweichung/Energie/Histogramme\
-> in verschiedenen Bändern (Radien) mit Magnitude der Fourier Transform

Bis nächstes Mal mit verschiedenen Filtern 

------------------------------------------------------------------------------------------------------

13.11.\
Bandpassfilter an Bilder anpassen (nicht quadratisch)\
Ob Abweichung passt erst bei größeren Experimenten sichtbar\
Mehr als drei Bänder\
Bis nächstes Mal in KNN -> und erste Ergebnisse\
Bei Featurevektor alles in einen Vektor, aber schauen, wie passt

------------------------------------------------------------------------------------------------------

27.11.\
KNN umschreiben (Scikit? -> LeaveOneOut)\
Energie statt Abweichung!!! (Summe von n² von Koeffizienten) im Band\
Synthetische einberechnen\
Genuine vs Spoof im Ergebnis\
Bis zum nächsten Mal: eine Datenbank mit synth. vergleichen

------------------------------------------------------------------------------------------------------

11.12.\
Bei quadratischen synthetischen Bildern Größe anpassen (Python Code von Prof)\
Scikit KNN!\
Nächster Schritt: LeaveOutOut -> LeaveOneSubjectOut/LeaveOnePersonOut Cross Validation
	-> Keine Bilder von dieser Person in Trainingsdaten!

1. Genuine vs Spoofed
2. Genuine vs Synthetisch\
Ziel: Genuine + Synthetisch -> Spoofed erkennen

Bis 08.01. alle Daten -> Auswertungen

------------------------------------------------------------------------------------------------------

08.01.\
Vollstaendige Ergebnisse\
Einfluss von Anzahl Bändern und k\
Präsentation bis zum naechsten Mal\

------------------------------------------------------------------------------------------------------

22.01.\
Balanciert -> Nur Genuine verwenden, die auch in synthetisch/spoofed sind (gleich viele)\
LeaveOneSubjectOut UND LeaveOneOut\
Schauen in welchen Baendern unterschiede (nicht alle 30 Baender verwenden) -> diese Baender rekonstruieren\
Ein Band rekonstruieren?\
Praesentation an Prof schicken (nicht erst am letzten Tag)















