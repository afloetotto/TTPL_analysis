import warnings
import numpy as np
from scipy import stats, constants


def kalibrier(peakpos, peakpos_vergleich, T=4.2, T_vergleich=4.2):
    '''
    Kalibrierung von TTPL-Spektren. Hierfür wird eine lineare Regression zwischen
    den vom Fit ermittelten Peakpositionen und den vom Nutzer angegebenen
    Vergleichswerten durchgeführt. Anschließend wird die Energie-Achse des
    Spektrums mithilfe dieser Geraden kalibriert.

    Parameter:
        peakpos : List
        Positionen der Peaks im Spektrum.

        peakpos_vergleich : List
        Vergleichswerte für die Positionen der Peaks.

        T : Float
        Probentemperatur bei der Messung des Spektrums.

        T_vergleich : List
        Probentemperaturen bei der Messung der Vergleichswerte. Die Liste muss
        genau so lang sein wie peakpos_vergleich. Falls für eine Peakposition
        kein Vergleichswert existiert kann an die Stelle np.nan eingetragen
        werden.

    Returns:
        Steigung : Float

        Y-Achsenabschnitt : Float
    '''
    peakpos = np.array(peakpos)
    peakpos_vergleich = np.array(peakpos_vergleich)
    T_vergleich = np.array(T_vergleich)

    # Sortiere Wertepaare mit einem NaN aus
    test = [not(np.isnan(ele)) for ele in peakpos_vergleich]
    peakpos = peakpos[test]
    peakpos_vergleich = peakpos_vergleich[test]
    T_vergleich = T_vergleich[test]

    # Benutze T - T_vergleich, um peakpos_vergleich zu korrigieren
    if T > 0:
        # Falls ITO nicht vorhanden -> T = -1
        if len(T_vergleich) != len(peakpos_vergleich):
            raise Exception(
                'T_vergleich und peakpos_vergleich sind unterschiedliche lang'
            )
        for i in range(len(peakpos_vergleich)):
            peakpos_vergleich[i] += constants.Boltzmann * (T - T_vergleich[i])\
            / 2 / constants.e

    # Berechne die lineare Regression
    reg = stats.linregress(peakpos, peakpos_vergleich)
    if reg.rvalue < 0.99:
        warnings.warn("Korrelation der linearen Regression zur Kalibrierung ist kleiner als 0.99")

    print('E-Kalibrierung Steigung:', reg.slope)
    # print('E-Kalibrierung Y-Achsenabschnitt:', reg.intercept)
    print('E-Kalibrierung Korrelation:', reg.rvalue)

    return reg.slope, reg.intercept


def kalibriere_iterativ(peakpos, peakpos_vergleich, T=4.2, T_vergleich=4.2):
    '''
    Problem:
    Wechselseitige Abhängigkeit zwischen der Temperatur und der
    Kalibriergeraden.
    Ansatz:
    Rechne mit der Starttemperatur eine Kalibriergerade aus und benutze diese
    anschließend um die Temperatur zu korrigieren (T_neu = Steigung * T). Mit
    der neuen Temperatur wird dann eine neue Kalibrierungsgerade berechnet.
    Abgebrochen wird sobald die relative Änderung der Temperatur zwischen zwei
    Iterationsschritten hinreichend klein ist.
    '''
    Tratio = 1.0
    while Tratio > 0.1:
        slope, intercept = kalibrier(peakpos, peakpos_vergleich, T=T,
                                     T_vergleich=T_vergleich)
        dmy = T
        T = T * slope
        Tratio = abs(T - dmy) / dmy

    return slope, intercept
