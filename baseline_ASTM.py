import numpy as np


def base(E, I, roi, fenster=0):
    '''
    Berechnet die Baseline des Spektrums so wie in der ASTM-Vorschrift gezeigt.
    Die Punkte, deren E-Werte in roi angegeben sind, werden mit Geraden
    verbunden. Die Baseline setzt sich aus diesen Geraden zusammen.

    Parameter:
    E : np-Array
      Energie-Werte des Spektrums
    I : np-Array
      Intensitäts-Werte des Spektrums
    roi : list
      Regions of interest: E-Werte der Punkte, die wie in der ASTM-Vorschrift
      beschrieben mit Geraden verbunden werden. Der erste und letzte Punkt wird
      automatisch als Stützstelle für eine Gerade benutzt, auch wenn er nicht
      in roi steht.
    fenster : Float
      Beispiel: fenster = 0.5 -> Es wird in einem Umkreis von 0.5 meV um die
      Energien in roi nach Intensitätsminima gesucht. Die Positionen der
      Minima werden als 'neues' roi genutzt.

    '''

    # Streiche alle Werte aus roi, die nicht im Energiebereich des Spektrums
    # liegen
    Emin = min(E)
    Emax = max(E)
    roi_neu = []
    for ele in roi:
        if ele > Emin and ele < Emax:
            roi_neu.append(ele)
    roi = roi_neu

    # Suche im angegebenen Fenster nach Minima
    if fenster > 0:
        fenster_ind = int(fenster / 1000 / abs(E[1] - E[0]))
        roi_neu = []
        for ele in roi:
            ind = index(E, ele)
            ind_neu = np.argmin(I[ind-fenster_ind:ind+fenster_ind]) +\
                    ind - fenster_ind
            roi_neu.append(E[ind_neu])
        roi = roi_neu

    # Füge die kleinste und größte Energie zu roi hinzu
    roi.append(Emin)
    roi.append(Emax)

    # Sortiere E aufsteigend. Falls E vorher absteigend sortiert war, muss base
    # nachher nochmal umsortiert werden, damit es zum ursprünglichen E passt.
    if E[0] > E[1]:
        # E ist absteigend sortiert
        umsortiert = True
        E = E[::-1]
        I = I[::-1]
    else:
        umsortiert = False

    # Sortiere roi und bestimme die Indizes, die die Werte aus roi in E haben.
    roi = np.sort(roi)
    roi_inds = [index(E, ele) for ele in roi]

    # Initialisiere die Baseline
    N = len(E)
    base = np.zeros(N)
    k = 0

    for i in range(len(roi) - 1):
        # Berechne die Parameter der Geraden y(x) = m * x + n
        x0 = E[roi_inds[i]]
        y0 = I[roi_inds[i]]
        x1 = E[roi_inds[i+1]]
        y1 = I[roi_inds[i+1]]
        m = (y1 - y0) / (x1 - x0)
        n = y0 - m * x0

        # Berechne die Baseline aus den Geraden-Parametern
        Nroi = roi_inds[i+1] - roi_inds[i]
        for j in range(Nroi):
            base[k] = m * E[k] + n
            k += 1
    base[N-1] = y1

    if umsortiert:
        base = base[::-1]
    return base


def index(arr, num):
    '''
    Sucht welche Zahl in arr dem Wert num am nächsten ist und gibt deren Index
    aus.
    '''

    diff = [abs(ele - num) for ele in arr]
    minval = np.amin(diff)
    for ind, val in enumerate(diff):
        if val == minval:
            return ind






















