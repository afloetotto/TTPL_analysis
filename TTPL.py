import baseline_ASTM
import TTPL_import
import TTPL_Kalibrierung
import csv
import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, optimize, constants


def plot_spektrum(datei):
    '''
    Importiert die Daten aus der angegebenen Datei und plottet das Spektrum.
    Im Plotfenster wird oben rechts die Position des Cursors angezeigt. Daher
    kann diese Funktion benutzt werden, um Parameter von TTPL_auswertung()
    oder TTPL_einzelnes_spektrum() herauszufinden.

    Parameter:
        datei : String
        Name/Pfad einer Messdatei (.asc von Andor Solis oder .xlsx vom
        LabView-Messprogramm)
    '''
    E, I = TTPL_import.read_file(datei)

    plt.style.use(['science', 'vibrant', 'no-latex'])
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(E, I)
    ax.set_xlabel('E [eV]')
    ax.set_ylabel('Intensität [b. E.]')
    plt.show()


def gauss(x, Amp, sigma, x0):
    '''
    Definition der 1D Gaußfunktion.

    Parameters:
        x : List
        Energiewerte des Spektrums.

        Amp : Float
        Maximaler Wert des Peaks.

        sigma : Float
        Halbwertsbreite des Maximums

        x0 : Float
        Position (Energie) des Maximums
    '''

    sigma_neu = sigma / np.sqrt(8 * np.log(2))  # Damit das eingebene sigma
    # der Halbwertsbreite entspricht

    return Amp * np.exp(-(x - x0)**2 / 2 / sigma_neu**2)


def lorentz(x, Amp, sigma, x0):
    '''
    Definition der 1D Lorentzfunktion.
    Beachte: Das hier ist eine genäherte Darstellung der Lorentzfunktion, in
    der die Halbwertsbreite leichter zu bestimmen ist (siehe Wikipedia)

    Parameters:
        x : List
        Energiewerte des Spektrums.

        Amp : Float
        Maximaler Wert des Peaks.

        sigma : Float
        Halbwertsbreite des Maximums

        x0 : Float
        Position (Energie) des Maximums
    '''

    Amp_neu = Amp * sigma**2 / 4  # Damit die eingebene Amp dem maximalen
    # Wert entspricht
    return Amp_neu / ((x - x0)**2 + sigma**2 / 4)


def maxwellboltzmann_quadr(x, Amp, sigma, x0):
    '''
    Maxwell-Boltzmann-Fitfunktion (quadratisch Anteil) aus den Büchern von
    Sauer und Smoliner.

    Parameters:
        x : List
        Energiewerte des Spektrums.

        Amp : Float
        Maximaler Wert des Peaks.

        sigma : Float
        Halbwertsbreite des Maximums

        x0 : Float
        Position (Energie) des Maximums
    '''

    sigma_neu = sigma / 3.4
    Amp_neu = Amp / 4 / sigma_neu**2 / np.exp(-2)
    x0_neu = x0 - 2 * sigma_neu

    n = len(x)
    werte = np.zeros(n)
    for i in range(n):
        if x[i] > x0_neu:
            werte[i] = Amp_neu * np.exp(
                -(x[i] - x0_neu) / sigma_neu) * (x[i] - x0_neu)**2
    return werte


def maxwellboltzmann_wurz(x, Amp, sigma, x0, sigma_faltung=-1):
    '''
    Maxwell-Boltzmann-Fitfunktion (wurzelförmiger Anteil). Die Temperatur kann
    mit sigma = k * T berechnet werden. k ist die Boltzmann-Konstante. Ggf.
    wird die Maxwell-Boltzmann-Verteilung mit einer Gaußfunktion gefaltet. So
    können die Spalt-Verbreitung und die Gaußsche-Verbreiterung berücksichtigt
    werden.

    Parameters:
        x : List
        Energiewerte des Spektrums.

        Amp : Float
        Maximaler Wert des Peaks.

        sigma : Float
        Es gilt sigma = k * T, wobei k die Boltzmann-Konstante ist.

        x0 : Float
        Position (Energie) des Maximums

        sigma_faltung : Float
        Falls sigma faltung > 0 ist wird die Maxwell-Boltzmann-Verteilung mit
        einer Gaußfunktion gefaltet. sigma_faltung entspricht der
        Standardabweichungen dieser Gaußfunktion.
    '''
    sigma = sigma / 1.795
    x0 = x0 - sigma / 2

    n = len(x)
    fitfunc = np.zeros(n)
    for i in range(n):
        if x[i] > x0:
            fitfunc[i] = np.exp(-(x[i] - x0) / sigma) * np.sqrt(x[i] - x0)

    if sigma_faltung <= 0:
        return fitfunc * Amp / max(fitfunc)
    else:
        delta_E = x[1] - x[0]
        faltfunc = signal.windows.gaussian(1000, sigma_faltung / delta_E)
        faltung = signal.convolve(fitfunc, faltfunc,
                                  mode='same') / sum(faltfunc)
        return faltung * Amp / max(faltung)


def SiPeaks(x, Ampver, sigma, x0, Delta, sigma_faltung):
    '''
    Mathematische beschreibung der beiden Silizium Peaks (ITO und ILO) angelegt
    an Pelant. Das Maximum der Summe von ITO und ILO ist auf 1 normiert.

    Die Parameter Delta (Abstand zwischen ITO und ILO) und sigma_faltung
    (Standardabweichung der Gaußfunktion für die Faltung) müssen außerhalb
    dieser Funktion definiert werden. Sie sind keine Parameter, da sie bei Fits
    üblicherweise nicht angepasst werden sollen.

    Parameter:
        x: Float
        x-Werte des Spektrums (Energie)

        Ampver: Float
        ILO/ITO-Intensitätsverhältnis

        sigma: Float
        Breite der beiden Peaks. Die Breite lässt sich aus der Temperatur
        berechnen und ist identisch für beide Peaks.

        x0 : Float
        Positionen des ITO-Peaks. Die Position des ILO-Peaks wird mit einem
        experimentell ermittelten Abstand der beiden Peaks berechnet. Da dieser
        Abstand während möglichen eines Fits nicht verändert werden soll ist er
        kein Argument für diese Funktion. Er heißt Delta und wird außerhalb der
        Funktion definiert.

    Return:
        Spektrum: List

        ITO: List

        ILO: List
    '''
    ITO = maxwellboltzmann_wurz(x, 1, sigma, x0, sigma_faltung=sigma_faltung)
    ILO = maxwellboltzmann_wurz(x,
                                Ampver,
                                sigma,
                                x0 + Delta,
                                sigma_faltung=sigma_faltung)
    Spektrum = ITO + ILO

    # Normiere das Maximum von Spektrum
    maxval = max(Spektrum)
    Spektrum /= maxval
    ITO /= maxval
    ILO /= maxval

    return Spektrum


def gebundene_exzitonen(x, Amp, sigma, x0, sigma_faltung):
    '''
    Fitfunktion für Peaks gebundener Exzitonen. Es handelt sich um eine
    Lorentzfunktion, die ggf. mit einer Gaußfunktion gefaltet wird.

    Parameters:
        x : List
        Energiewerte des Spektrums.

        Amp : Float
        Maximaler Wert des Peaks.

        sigma : Float
        Halbwertsbreite des Peaks

        x0 : Float
        Position (Energie) des Maximums

        sigma_faltung : Float
        Falls sigma faltung > 0 ist wird die Lozentzfunktion mit
        einer Gaußfunktion gefaltet. sigma_faltung entspricht der
        Standardabweichungen dieser Gaußfunktion.
    '''
    fitfunc = lorentz(x, Amp, sigma, x0)

    if sigma_faltung <= 0:
        return fitfunc * Amp / max(fitfunc)
    else:
        delta_E = x[1] - x[0]
        faltfunc = signal.windows.gaussian(1000, sigma_faltung / delta_E)
        faltung = signal.convolve(fitfunc, faltfunc,
                                  mode='same') / sum(faltfunc)
        return faltung * Amp / max(faltung)


def import_spektrum(datei, Npeaks, ITO_pos, roi, basekorr):
    '''
    Import ein Spektrum aus einer Messdatei und berechnet die Baselinekorrektur
    sowie die Normierung auf den ITO-Peak

    Parameter:
        datei: String
        Name/Pfad der Messdatei

        Npeaks: Integer
        Anzahl der Peaks im Spektrum

        ITO_pos: Float
        Energie des ITO-Peaks.

        roi: List
        Position in eV, die bei der Baselinekorrektur mit Geraden verbunden
        sollen (Siehe docstring sowie baseline_ASTM.base() sowie die
        ASTM-Vorschrift zu TTPL-Spektroskopie)

        basekorr: Boolean
        Entscheidet, ob eine Baselinekorrektur durchgeführt wird
    '''

    # Importiere die Daten
    E, I = TTPL_import.read_file(datei)

    # Entferne die Baseline
    if basekorr:
        base = baseline_ASTM.base(E, I, roi)
        I = I - base

    # Suche die Peaks
    peaks_alle, properties = signal.find_peaks(I, prominence=50)
    if len(peaks_alle) < Npeaks:
        Npeaks = len(peaks_alle)
        warnings.warn(datei + ': Das Spektrum enthält weniger Peaks als\
                      angegeben. Gebe weniger Peaks vor oder ändere das\
                      Argument prominence von signal.find_peaks()')
    reihenfolge = np.argsort(properties['prominences'])[::-1]
    peaks = peaks_alle[reihenfolge][0:Npeaks]
    peaks = np.sort(peaks)[::-1]

    # Normiere das Signal auf die Höhe des ITO-Peaks
    ITO_index = baseline_ASTM.index(E[peaks], ITO_pos)
    I = I / I[peaks[ITO_index]]

    return E, I


def einzelnes_spektrum(datei,
                       peakarten,
                       ITO_pos,
                       BTO_pos,
                       roi,
                       P_pos,
                       benutze_fit,
                       smooth=(9, 6),
                       x0_diff=1,
                       x0_puffer=1,
                       peakpos=None,
                       Delta=2e-3,
                       peakpos_vergleich=None,
                       T_vergleich=None,
                       export=False,
                       roi_fenster=0,
                       ILO_ratio_grenzen=(0, 1),
                       sigma_faltung_interval=(8e-5, 3e-4),
                       E_int=(-np.inf, np.inf),
                       T_start=25):
    '''
    Wertet ein TTPL-Spektrum aus. Das Spektrum und die Fitfunktion werden
    gezeichnet und in einer pdf-Datei gespeichert. Diese pdf-Datei hat den
    selben Namen wie die Messdatei und liegt im selben Ordner. Außerdem werden
    wichtige Eigenschaften des Spektrums berechnet und ausgegeben.

    Parameter:
        datei : String
        Name/Pfad der Messdatei. Es werden asc-Dateien von Andor Solis und
        xlsx-Dateien vom LabView-Messprogramm unterstützt (xls-Dateien müssen
        erst als xlsx-Datei gespeichert werden).

        peakarten : String
        Legt fest welche Funktionen zum Fitten der einzelnen Spektren benutzt
        werden sollen. Z.B. peakarten= 'LGW' für ein Spektrum mit drei Peaks,
        dessen Peaks mit einem Lorentz-, einem Gauß- und einer
        Maxwell-Boltzmann-Funktion mit wurzelförmigem Anteil gefitted werden
        sollen. Sortiert wird aufsteigend nach der Energie. Die Möglichkeiten
        sind: L (Lorentz), G (Gauß), Q (Maxwell-Boltzmann mit quadratischem
        Anteil), W (Maxwell-Boltzmann mit wurzelförmigem Anteil), S (Beschreibt
        ITO und ILO wie im Paper von Pelant: Je eine Faltung von W mit G), B
        (Bound-Exciton-Peak; Faltung aus Lorentz und Gauß)

        ITO_pos : Float
        Position des ITO-Peaks in eV.

        BTO_pos : Float
        Position des BTO-Peaks in eV.

        roi : List
        Position in eV, die bei der Baselinekorrektur mit Geraden verbunden
        sollen (Siehe docstring sowie baseline_ASTM.base() sowie die
        ASTM-Vorschrift zu TTPL-Spektroskopie)

        P_pos : Float
        Position der P-Linie in eV. Falls das Spektrum keine P-Linie enthält
        kann -1 angegebn werden.

        benutze_fit : Bool
        Entscheidet, ob die Werte der Fitfunktion zur Berechnung der
        Intensitätsverhältnisse benutzte werden. Falls nicht werden lokale
        Maxima des gemessenen Spektrums benutzt.

        smooth : 2-Tuple
        Fensterbreite und Polynomordnung für einen Savitzky-Golay-Filter der
        Intensitäten. Falls smooth = None wird kein Filter
        angewendet.

        x0_diff : Float
        Bestimmt die Grenzen beim fit für die Peakpositionen gelten. Bei
        x0_diff = 1 dürfen die Peakpositionen beim Fit in beide Richtungen
        um bis zu 1 meV verändert werden. Um die Peakpositionen während des
        Fits effektiv nicht anzupassen, kann hier ein sehr kleiner Wert
        eingetragen werden.

        peakpos : List
        Vorgabe für die Positionen der Peaks. Bei peakpos=None werden die Peaks
        mit scipy.signal.find_peaks() erkannt. Falls bei peakpos!=None in einem
        Interval von +- x0_puffer um die vorgegebenen Werte von
        scipy.signal.find_peaks() Peaks erkannt werden, werden deren Positionen
        benutzt.

        x0_puffer : Float
        Siehe beschreibung von peakpos. Angabe ist in meV.

        Delta : Float
        Abstand zwischen ITO und ILO. Wird für die Anpassung des Si-Peaks aus
        ITO und ILO benötigt.

        sigma_faltung : Float
        Wird für die Anpassung des Si-Peaks aus ITO und ILO benötigt.
        Halbwertsbreite der Gaußfunktionen, die wie von Pelant beschrieben mit
        den Maxwell-Boltzmann-Funktionen (Wurzel) gefaltet werden.

        peakpos_vergleich : List
        Vergleichswerte für die Positionen der Peaks für der Kalibrierung der
        Energie-Achse.

        T_vergleich : Float
        Probentemperaturen bei der Messung der Vergleichswerte. Die Liste muss
        genau so lang sein wie peakpos_vergleich. Achtung: Möglicherweise ist
        in der Literatur nicht die Probentemperatur sondern die Badtemperatur
        angegeben.

        export : Boolean
        Bestimmt ob die Daten (E-Achse, Intensität, Fitfunktion, Einzelne
        Bestandteile der Fitfunktion) als Textdatei importiert werden sollen.

        roi_fenster : Float
        Wird für die Baselinekorrektur benötigt. Beispiel: roi_fenster = 0.5 ->
        Es wird in einem Umkreis von 0.5 meV um die Energien in roi nach
        Intensitätsminima
        gesucht. Die Positionen der Minima werden als 'neues' roi genutzt.

        ILO_ratio_grenzen : 2-Tuple
        Grenzen für das ILO/ITO Peakverhältnis beim Pelant-Fit der beiden
        intrinsischen Peaks. Bei vergleichsweise hohen Temperaturen neigt der
        Fit dazu eine zu kleine Temperatur zu wählen und das mit einem zu hohen
        Peakverhältnis auszugleichen. In diesem Fall sollte ILO_ratio_max
        verringert werden. Andernfalls kann getrost ein Wert von 1 benutzt
        werden.

        sigma_faltung_interval : 2-Tuple
        sigma_faltung ist die Standardabweichung der Gaussfunktion, die wie von
        Pelant beschrieben mit dem Si-Peaks (und den BE-Peaks) gefaltet wird.
        Dieser Parameter entspricht dem Interval, in dem sigma_faltung beim Fit
        variiert werden kann. Der Mittelwert des Intervals wir als Startwert
        benutzt.

        E_int : 2-Tuple
        Interval für die Energieachse. Alle Werte außerhalb des Intervals
        werden abgeschnitten.

        T_start : Float
        Temperatur, die den Startwert für die Breite des intrinsischen Peaks
        beim Fit bestimmt.

        Returns:
            ToDo
    '''
    print('Beginne Auswertung von:', datei)

    # Rechne von meV in eV um
    x0_diff *= 1e-3
    x0_puffer *= 1e-3

    # Importiere die Daten
    E, I = TTPL_import.read_file(datei, E_int=E_int)

    # Entferne die Baseline
    base = baseline_ASTM.base(E, I, roi, fenster=roi_fenster)
    I = I - base

    # Glätte das Spektrum
    if smooth != None:
        I = signal.savgol_filter(I, smooth[0], smooth[1])

    # Suche die Peaks
    Npeaks = len(peakarten)
    peaks_alle, properties = signal.find_peaks(I, prominence=10)
    peakproms_alle = properties['prominences']
    if len(peakproms_alle) < Npeaks:
        warnings.warn(
            datei +
            " Es wurden weniger Peaks gefunden als die Anzahl vorgegebener Peaks. Evtl. muss das Argument prominence von scipy.signal.find_peaks() angepasst werden."
        )

    if peakpos != None:
        # Hat signal.find_peaks() in der Nähe der Vorgaben Peaks gefunden?
        if len(peakpos) != Npeaks:
            raise Exception(
                "Laengen von peakpos und peakarten stimmen nicht ueberein.")
        peaks = []
        peakproms = []
        order = np.argsort(peakpos)
        peakpos = [peakpos[ele] for ele in order]
        peakpos_vergleich = [peakpos_vergleich[ele] for ele in order]
        T_vergleich = [T_vergleich[ele] for ele in order]
        peakpos_vergleich_temp = []
        T_vergleich_temp = []
        peakarten_temp = []
        i = 0
        for ele in peakpos:
            dists = [abs(ele - E[peak]) for peak in peaks_alle]
            test = [dist <= x0_puffer for dist in dists]
            if np.any(test):
                proms = peakproms_alle[test]
                maxprom = np.max(proms)
                index = baseline_ASTM.index(peakproms_alle, maxprom)
                peaks.append(peaks_alle[index])
                peakproms.append(peakproms_alle[index])
                peakarten_temp.append(peakarten[i])
                peakpos_vergleich_temp.append(peakpos_vergleich[i])
                T_vergleich_temp.append(T_vergleich[i])
            i += 1

        if len(peaks) == 0:
            raise Exception("Nahe der angegebenen Werte wurden keine Peaks\
                            gefunden -> Korrigiere die vorgegebenen\
                            Peakpositionen oder erhoehe x0_puffer")
            peakpos = None
        else:
            peakpos_vergleich = peakpos_vergleich_temp
            T_vergleich = T_vergleich_temp
            peakarten = peakarten_temp

    if np.all(peakpos == None):
        # Benutze die Positionen von signal.find_peaks()
        reihenfolge = np.argsort(peakproms_alle)[::-1]
        peakproms = peakproms_alle[reihenfolge][0:Npeaks]
        peaks = peaks_alle[reihenfolge][0:Npeaks]
        reihenfolge2 = np.argsort(peaks)[::-1]
        peakproms = peakproms[reihenfolge2]
        peaks = peaks[reihenfolge2]

    # Berechne die Peakbreiten (um sie für die Startparameter des Fits zu
    # benutzten)
    Npeaks = len(peaks)
    peakbreiten = signal.peak_widths(I, peaks)[0] * (E[1] - E[0])
    peakbreiten = np.array([abs(ele) for ele in peakbreiten])

    # Bestimme die Indizes des ITO- und des BTO-Peaks
    ITO_index = baseline_ASTM.index(E[peaks], ITO_pos)
    BTO_index = baseline_ASTM.index(E[peaks], BTO_pos)
    P_index = baseline_ASTM.index(E[peaks], P_pos)

    # Berechne T aus der FWHM für einen Vergleich mit dem Fit-Ergebniss
    T_FWHM = peakbreiten[ITO_index] / constants.k / 1.8 * constants.e

    # Normiere das Signal auf die Höhe des ITO-Peaks
    ITO_Peakheight_roh = I[peaks[ITO_index]]
    I = I / ITO_Peakheight_roh
    peakproms = peakproms / peakproms[ITO_index]

    # Bestimme die Anfangsparameter für den Fit
    Amp_start = peakproms.copy()
    sigma_start = peakbreiten.copy()
    x0_start = E[peaks].copy()

    for i in range(len(peaks)):
        if peakarten[i] == 'S':
            # Gebe Anfangswert für das ITO/ILO Peakverhältnis vor
            Amp_start[i] = (ILO_ratio_grenzen[1] + ILO_ratio_grenzen[0]) / 2
            # Gebe den Anfangswert für die Temperatur vor
            sigma_start[i] = 3.4 * T_start * constants.Boltzmann / constants.e

    startparams = np.array([])
    startparams = np.append(startparams, Amp_start)
    startparams = np.append(startparams, sigma_start)
    startparams = np.append(startparams, x0_start)
    if 'S' in peakarten or 'B' in peakarten:
        startparams = np.append(startparams, np.mean(sigma_faltung_interval))
    startparams.tolist()

    # Definiere die Fitfunktion
    def fitfunc(x, *params):
        '''
        Fitfunktion des Spektrums. Jeder einzelne Peak wird durch eine Funktion
        dargestellt. Das Spektrum ist die Summe dieser Funktionen.

        Parameter:   Amps : sequence
                          Amplituden der Gaußfunktionen.
                      sigmas : sequence
                          Standardabweichungen der Gaußfunktionen.
                      x0s : sequence
                          Verschiebungen entlang der Abzisse der
                          Gaußfunktionen.
        '''

        if 'S' in peakarten or 'B' in peakarten:
            sigma_faltung = params[-1]
            num = (len(params) - 1) / 3
        else:
            num = len(params) / 3

        if num != np.round(num):
            raise Exception(
                'Anzahl der params muss durch 3 teilbar sein (siehe docstring)'
            )
        else:
            Npeaks = int(num)

        wert = 0
        for i in range(Npeaks):
            A = params[i]
            sigma = params[i + Npeaks]
            x0 = params[i + 2 * Npeaks]
            if peakarten[i] == 'G':
                # gaußförmiger Peak
                wert += gauss(x, A, sigma, x0)
            elif peakarten[i] == 'Q':
                # Maxwell-Boltzmann Peak mit quadratischem Anteil
                wert += maxwellboltzmann_quadr(x, A, sigma, x0)
            elif peakarten[i] == 'W':
                # Maxwell-Boltzmann Peak mit wurzelförmigem Anteil
                wert += maxwellboltzmann_wurz(x, A, sigma, x0)
            elif peakarten[i] == 'S':
                # Silizium Peak wie bei Pelant
                wert += SiPeaks(x, A, sigma, x0, Delta, sigma_faltung)
            elif peakarten[i] == 'L':
                # lorentzförmiger Peak
                wert += lorentz(x, A, sigma, x0)
            elif peakarten[i] == 'B':
                # Bound-Exciton-Peak
                wert += gebundene_exzitonen(x, A, sigma, x0, sigma_faltung)
            else:
                raise Exception(
                    'Nur die folgenden peakarten sind verfuegbar: G, L, Q, W, B (siehe Docstring)'
                )

        return wert

    # Lege die Grenze der Parameter während des Fits fest und formatiere sie so
    # wie in der Doku von scipy.optimize.curve_fit() beschrieben
    grenzen_min = np.zeros(Npeaks)  # Amplitude
    grenzen_max = np.array([np.inf] * Npeaks)

    ILO_ratio_grenzen = np.sort(ILO_ratio_grenzen)
    for i in range(len(peakarten)):
        if peakarten[i] == 'S':
            grenzen_min[i] = ILO_ratio_grenzen[0]
            grenzen_max[i] = ILO_ratio_grenzen[1]

    grenzen_min = np.append(grenzen_min, [1e-6] * Npeaks)  # Sigma
    grenzen_max = np.append(grenzen_max, (3 * sigma_start).tolist())
    grenzen_min = np.append(grenzen_min,
                            [pos - x0_diff for pos in x0_start])  # x0
    grenzen_max = np.append(grenzen_max, [pos + x0_diff for pos in x0_start])
    if 'S' in peakarten or 'B' in peakarten:
        # Grenzen für sigma_faltung
        sigma_faltung_interval = np.sort(sigma_faltung_interval)
        grenzen_min = np.append(grenzen_min, sigma_faltung_interval[0])
        grenzen_max = np.append(grenzen_max, sigma_faltung_interval[1])

    grenzen = (grenzen_min, grenzen_max)

    # Fitte das Spektrum
    popt, pcov = optimize.curve_fit(fitfunc,
                                    E,
                                    I,
                                    p0=startparams,
                                    bounds=grenzen,
                                    maxfev=100000)
    # popt = startparams
    # pcov = np.zeros((len(startparams), len(startparams)))

    perr = np.sqrt(np.diag(pcov))  # Fehler der Parameter
    Amps = popt[0:Npeaks]
    sigmas = popt[Npeaks:2 * Npeaks]
    sigmas_err = perr[Npeaks:2 * Npeaks]
    sigmas = [abs(val) for val in sigmas]
    x0s = popt[2 * Npeaks:3 * Npeaks]
    if 'S' in peakarten or 'B' in peakarten:
        sigma_faltung = popt[-1]
        print('sigma_faltung:', sigma_faltung)

    # Sortiere die Parameter nach der Energie, falls während des Fits Peaks
    # vertauscht wurden.
    order = np.argsort(x0s)
    Amps = [Amps[ele] for ele in order]
    x0s = [x0s[ele] for ele in order]
    sigmas = [sigmas[ele] for ele in order]

    # Berechne die Funktionswerte der Fitfunktion
    fitvals = fitfunc(E, *popt)

    # Ordne die Peaks zu
    puffer = 0.002
    if x0s[ITO_index] - ITO_pos <= puffer:
        ITO_Linie = True
    else:
        ITO_Linie = False
        warnings.warn(
            datei +
            ": Keine ITO-Linie nahe der angegebenen Position gefunden. Da die Temperatur so nicht aus dem Spektrum bestimmt werden kann, kann peakpos_vergleich vor der Kalibrierung nicht auf die richtige Temperatur umgerechnet werden."
        )

    if x0s[BTO_index] - BTO_pos <= puffer:
        BTO_Linie = True
    else:
        BTO_Linie = False

    if ITO_Linie and ITO_index < len(peaks) - 1:
        ILO_Linie = True
    else:
        ILO_Linie = False

    if x0s[P_index] - P_pos <= puffer:
        P_Linie = True
    else:
        P_Linie = False

    # Berechne weitere Eigenschaften des Spektrums
    print('ILO-Verhaeltnis:', Amps[ITO_index])
    if benutze_fit:
        if ITO_Linie and BTO_Linie:
            Intver = fitvals[peaks[BTO_index]] / fitvals[peaks[ITO_index]]
        else:
            Intver = 0

        if ITO_Linie:
            ITO_Halbwertsbreite = sigmas[ITO_index]
            if peakarten[ITO_index] in ['S', 'Q', 'W']:
                T = ITO_Halbwertsbreite / 3.4 / constants.Boltzmann * constants.e
                Terr = sigmas_err[
                    ITO_index] / 3.4 / constants.Boltzmann * constants.e
            else:
                T = ITO_Halbwertsbreite / 1.8 / constants.Boltzmann * constants.e
                Terr = sigmas_err[
                    ITO_index] / 1.8 / constants.Boltzmann * constants.e
        else:
            ITO_Halbwertsbreite = 0
            T = -1

        if BTO_Linie:
            BTO_Halbwertsbreite = sigmas[BTO_index]
        else:
            BTO_Halbwertsbreite = 0

        if ILO_Linie:
            ILO_Halbwertsbreite = sigmas[ITO_index + 1]
        else:
            ILO_Halbwertsbreite = 0

        if P_Linie:
            P_Intvers = fitvals[peaks[P_index]] / fitvals[peaks[ITO_index]]
        else:
            P_Intvers = 0
    else:
        if ITO_Linie and BTO_Linie:
            Intver = I[peaks[BTO_index]]
        else:
            Intver = 0

        if ITO_Linie:
            ITO_Halbwertsbreite = abs(peakbreiten[ITO_index])
            if peakarten[ITO_index] in ['S', 'Q', 'W']:
                ITO_Halbwertsbreite_fit = sigmas[ITO_index]
                T = sigmas[ITO_index] / 1.795 / constants.Boltzmann * constants.e
                Terr = sigmas_err[
                    ITO_index] / 1.795 / constants.Boltzmann * constants.e
            else:
                T = ITO_Halbwertsbreite / 1.795 / constants.Boltzmann * constants.e
                Terr = sigmas_err[
                    ITO_index] / 1.795 / constants.Boltzmann * constants.e
        else:
            ITO_Halbwertsbreite = 0
            T = -1
            Terr = -1

        if BTO_Linie:
            BTO_Halbwertsbreite = abs(peakbreiten[BTO_index])
        else:
            BTO_Halbwertsbreite = 0

        if ILO_Linie:
            ILO_Halbwertsbreite = peakbreiten[ITO_index + 1]
        else:
            ILO_Halbwertsbreite = 0

        if P_Linie:
            P_Intvers = I[peaks[P_index]]
        else:
            P_Intvers = 0

    # Kalibriere die Energie-Achse des Spektrums mithilfe der Vergleichswerte
    ########### WICHTIG: nach der Kalibrierung müssen die Ergebnisse evtl. korrigiert werden ##############
    if len(peakpos_vergleich) > 1:
        slope, intercept = TTPL_Kalibrierung.kalibrier(x0s,
                                                       peakpos_vergleich,
                                                       T=T,
                                                       T_vergleich=T_vergleich)
        E = [slope * ele + intercept for ele in E]
        x0s = [slope * ele + intercept for ele in x0s]
        sigmas = [slope * ele for ele in sigmas]
        ITO_Halbwertsbreite *= slope
        ITO_Halbwertsbreite_fit *= slope
        Delta *= slope
        T *= slope
        T_FWHM *= slope
        sigma_faltung *= slope
        if ITO_Linie: x0_ITO = x0s[ITO_index]
        else: x0_ITO = -1
        if BTO_Linie: x0_BTO = x0s[BTO_index]
        else: x0_BTO = -1
    else:
        warnings.warn(
            datei +
            ': Es konnte keine Kalibrierung durchgefuehrt werden, weil nicht mindestens 2 Peaks nahe der angegeben Werte gefunden wurden.'
        )

    # Vergleiche das T aus dem Fit mit dem T aus der FWHM
    abweichung = 10  # in %
    fehler = abs(T / T_FWHM - 1) * 100
    if fehler > abweichung:
        warnings.warn(datei + """
            : Die Temperatur vom Fit weicht mehr als %d%% (%.2f%%)
            von der Temperatur aus der Halbwertsbreite ab.
            """ % (abweichung, fehler))
    if T < 4.2:
        warnings.warn(datei + """
        : Die Probentemperatur vom Fit ist kleiner als 4.2 K. Da stimmt was
        nicht!
        """)

    print('T =', T)
    print('T aus Halbwertsbreite =', T_FWHM)

    # Plotte das normierte Spektrum
    # wl = [constants.h * constants.c / ele / constants.e * 1e9 for ele in E]
    plt.style.use(['science', 'vibrant'])
    plt.style.use('no-latex')
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_axes([0.08, 0.48, 0.9, 0.5])
    ax.plot(E, I, label='Messung', linewidth=2)
    ax.set_title(datei)
    ax.set_xlabel('Energie [eV]')
    ax.set_ylabel('Intensität [b. E.]')

    # Ergänze die Fit-Funktion zum Plot und bestimme Peakhöhen/-halbwertsbreiten
    lbreite = 1.5
    ax.plot(E, fitvals, '--', label='Fitfunktion', lw=lbreite)
    ax2 = fig.add_axes([0.08, 0.08, 0.9, 0.32])
    ax2.set_xlabel('Energie [eV]')
    ax2.set_ylabel('Intensität [b. E.]')
    fitvals_einzeln = []

    for i in range(Npeaks):
        if peakarten[i] == 'G':
            # gaußförmiger Peak
            vals = gauss(E, Amps[i], sigmas[i], x0s[i])
            fitvals_einzeln.append(vals)
            ax2.plot(E, vals, '-', label='Gauß', lw=lbreite)
        elif peakarten[i] == 'Q':
            # Maxwell-Boltzmann Peak mit quadratischem Anteil
            vals = maxwellboltzmann_quadr(E, Amps[i], sigmas[i], x0s[i])
            fitvals_einzeln.append(vals)
            ax2.plot(E,
                     vals,
                     '-.',
                     label='Maxwell-Boltzmann (Quadrat)',
                     lw=lbreite)
        elif peakarten[i] == 'W':
            # Maxwell-Boltzmann Peak mit wurzelförmigem Anteil
            vals = maxwellboltzmann_wurz(E, Amps[i], sigmas[i], x0s[i])
            fitvals_einzeln.append(vals)
            ax2.plot(E,
                     vals,
                     '-.',
                     label='Maxwell-Boltzmann (Wurzel)',
                     lw=lbreite)
        elif peakarten[i] == 'S':
            # Silizium Peak wie bei Pelant
            vals = SiPeaks(E, Amps[i], sigmas[i], x0s[i], Delta, sigma_faltung)
            fitvals_einzeln.append(vals)
            ax2.plot(E, vals, '-.', label='Si-Peaks (%.1f K)' % T, lw=lbreite)
        elif peakarten[i] == 'L':
            # lorentzförmiger Peak
            vals = lorentz(E, Amps[i], sigmas[i], x0s[i])
            fitvals_einzeln.append(vals)
            ax2.plot(E, vals, '-', label='Lorentz', lw=lbreite)
        elif peakarten[i] == 'B':
            # Bound-Exciton-Peak
            vals = gebundene_exzitonen(E, Amps[i], sigmas[i], x0s[i],
                                       sigma_faltung)
            fitvals_einzeln.append(vals)
            ax2.plot(E, vals, '-', label='BE-Peak', lw=lbreite)
    ax.legend()
    ax2.legend()

    # Kennzeichne die Peaks
    # if ITO_Linie:
    # ax.annotate('ITO', xy=(x0s[ITO_index], fitvals[peaks[ITO_index]]))

    # if BTO_Linie:
    # ax.annotate('BTO', xy=(x0s[BTO_index], fitvals[peaks[BTO_index]]))

    # if ILO_Linie:
    # ax.annotate('ILO', xy=(x0s[ITO_index+1], fitvals[peaks[ITO_index+1]]))

    # if P_Linie:
    # ax.annotate('P-Linie', xy=(x0s[P_index], fitvals[peaks[P_index]]))

    # Speichere den Plot
    fname = os.path.splitext(datei)[0]
    plt.savefig(fname + '.pdf')
    # plt.show()
    plt.close(fig)

    # Exportiere ggf. die Daten als Textdatei
    if export:
        df = pd.DataFrame()
        df['Energie'] = E
        df['Intensität'] = I
        df['Fitfunktion'] = fitvals
        for i in range(len(fitvals_einzeln)):
            name = 'Fit einzelner Peak ' + str(i + 1)
            df[name] = fitvals_einzeln[i]
        df['Temperatur'] = [T] * len(E)
        df.to_csv(fname + '_Daten.csv', index=False)

    return Intver, ITO_Peakheight_roh, sigma_faltung, P_Intvers, ITO_Halbwertsbreite, T, Terr, x0_BTO, x0_ITO, ITO_Halbwertsbreite_fit


def test_filetype(ftype):
    '''
    Testet, ob eine Datei die Endung einer Messdatei hat.
    Das LabView Programm speichert die Messung in einer .xls Datei, die
    dann noch in eine .xlsx Datei umgewandelt werden muss damit man sie
    importieren kann (Die Funktion pandas.read_excel() unterstützt zwar
    auch eine engine für .xls Dateien. Bei diesen Dateien kommt dann
    allerdings die Fehlermeldung "File is corrupted").

    Andor Solis speichert die Messungen als .asc Datei.
    '''

    if ftype == '.xlsx' or ftype == '.asc':
        return True
    else:
        return False


def auswertung(ordner, parameter):
    '''
    Wertet alle Messdateien in einem Ordner aus. Es wird ein neuer Ordner mit
    den Ergebnissen der Auswertung erstellt. Dazu gehören Diagramme der
    Spektren und Fitfunktionen sowie eine Tabelle mit den Eigenschaften der
    Spektren als csv-Datei.

    Wenn der intrinsische Silizium Doppelpeak wie von Pelant beschrieben
    angefittet wird (S im Parameter peakarten), kann die Probentemperatur mit T
    = ITO_Halbwertsbreite * k / e berechnet werden.

    Beachte: Die pdf-Dateien mit Diagrammen sowie die csv-Datei dürfen nicht
    geöffnet sein, wenn dieses Programm auf ihren Ordner losgelassen wird.
    Sonst kann das Programm nicht auf die Dateien zugreifen und verursacht eine
    Fehlermeldung.

    Parameter:
        ordner : String
        Name/Pfad eines Ordners, in dem sich ascii-Messdateien befinden.

        Parameter:
        parameter : Dict
        Dictionary mit den Parametern für TTPL.einzelnes_spektrum(). Zusätzlich
        hat dieses Dictionary ein Feld namens x0_fest. Dies ist ein Boolean,
        der bestimmt ob die Peakposition beim Fit angepasst werden sollen.
        Falls x0_fest = False werden die Peakposition nur mit
        scipy.signal.find_peaks() bestimmt.
    '''

    # Erstelle eine Liste mit allen Dateien im Ordner
    dirlist = os.listdir(ordner)
    dateien = np.array([
        f for f in dirlist if os.path.isfile(os.path.join(ordner, f))
        and test_filetype(os.path.splitext(f)[1])
    ])

    # Wechsle in den Ordner
    ordner_urspruenglich = os.getcwd()
    os.chdir(ordner)

    # Sortiere Dateien aus, die nicht ausgewertet werden sollen
    # dateien_auswahl = []
    # for ele in dateien:
    # # Sortiere Messungen mit 150er oder 600er Gitter aus
    # test1 = not(re.search('_\d\d\d.asc', ele))
    # # Sortiere Messungen an W1 und W2 aus
    # test2 = not(re.search('W1_', ele))
    # test3 = not(re.search('W2_', ele))
    # if test1 and test2 and test3:
    # dateien_auswahl.append(ele)
    # dateien = dateien_auswahl
    # warnings.warn('Es werden nur bestimmte Messdateien ausgewertet. Siehe TTPL.py fuer die Kriterien')

    # Werte die Dateien aus
    peakarten = parameter['peakarten']
    ITO_pos = parameter['ITO_pos']
    BTO_pos = parameter['BTO_pos']
    roi = parameter['roi']
    P_pos = parameter['P_pos']
    benutze_fit = parameter['benutze_fit']
    smooth = parameter['smooth']
    x0_diff = parameter['x0_diff']
    x0_puffer = parameter['x0_puffer']
    peakpos = parameter['peakpos']
    Delta = parameter['Delta']
    peakpos_vergleich = parameter['peakpos_vergleich']
    T_vergleich = parameter['T_vergleich']
    export = parameter['export']
    roi_fenster = parameter['roi_fenster']
    ILO_ratio_grenzen = parameter['ILO_ratio_grenzen']
    sigma_faltung_interval = parameter['sigma_faltung_interval']
    E_int = parameter['E_int']
    T_start = parameter['T_start']

    n = len(dateien)
    Intvers = np.zeros(n)
    ITO_FWHMs = np.zeros(n)
    sigma_faltungs = np.zeros(n)
    P_Intvers = np.zeros(n)
    ITO_Halbwertsbreite = np.zeros(n)
    T = np.zeros(n)
    Terr = np.zeros(n)
    x0_BTO = np.zeros(n)
    x0_ITO = np.zeros(n)
    ITO_Halbwertsbreite_fit = np.zeros(n)

    for i in range(n):
        dmy = einzelnes_spektrum(dateien[i],
                                 peakarten,
                                 ITO_pos,
                                 BTO_pos,
                                 roi,
                                 P_pos,
                                 benutze_fit,
                                 smooth=smooth,
                                 x0_diff=x0_diff,
                                 x0_puffer=x0_puffer,
                                 peakpos=peakpos,
                                 Delta=Delta,
                                 peakpos_vergleich=peakpos_vergleich,
                                 T_vergleich=T_vergleich,
                                 export=export,
                                 roi_fenster=roi_fenster,
                                 ILO_ratio_grenzen=ILO_ratio_grenzen,
                                 sigma_faltung_interval=sigma_faltung_interval,
                                 E_int=E_int,
                                 T_start=T_start)
        Intvers[i] = dmy[0]
        ITO_FWHMs[i] = dmy[1]
        sigma_faltungs[i] = dmy[2]
        P_Intvers[i] = dmy[3]
        ITO_Halbwertsbreite[i] = dmy[4]
        T[i] = dmy[5]
        Terr[i] = dmy[6]
        x0_BTO[i] = dmy[7]
        x0_ITO[i] = dmy[8]
        ITO_Halbwertsbreite_fit[i] = dmy[9]

    # Speichere die Ergebnisse in einer csv-Datei:
    AuswertungFname = os.path.basename(ordner) + '_Auswertung.csv'
    f = open(AuswertungFname, 'w')
    writer = csv.writer(f, lineterminator='\n', delimiter=';')
    writer.writerow([
        'Datei', 'BTO/ITO Intensitätsverhältnis', 'ITO Höhe roh',
        'sigma Faltung', 'P/ITO Intensitätsverhältnis', 'ITO Halbwertsbreite',
        'T', 'Delta T', 'BTO_pos', 'ITO_pos', 'ITO Halbwertsbreite fit'
    ])
    for i in range(len(dateien)):
        writer.writerow([
            dateien[i], Intvers[i], ITO_FWHMs[i], sigma_faltungs[i],
            P_Intvers[i], ITO_Halbwertsbreite[i], T[i], Terr[i], x0_BTO[i],
            x0_ITO[i], ITO_Halbwertsbreite_fit[i]
        ])
    f.close()

    # Wechsle zurück in den ursprünglichen Ordner
    os.chdir(ordner_urspruenglich)

    print('Diagramme erstellt und als pdf-Dateien gespeichert.')
    print('Ergebnisse in', AuswertungFname, 'gespeichert.')
