import csv
import os
import numpy as np
import pandas as pd
from scipy import constants


def read_excel_file(datei):
    '''
    Liest eine Excel-Messdatei aus, die mit dem LabView-Programm erstellt
    wurde.

    Parameter:
        datei : String
        Name/Pfad der Messdatei

    Ausgaben:
        E : numpy Array
        Energiewerte des Spektrums

        I : numpy Array
        Intensitätswerte des Spektrums
    '''

    df = pd.read_excel(datei, skiprows=28, engine="openpyxl", header=None,
                       dtype=np.float, usecols='A:B',
                       names=['Wellenlaenge', 'Betrag'])
    wl = np.array(df.Wellenlaenge) * 1e-6
    I = np.array(df.Betrag)

    # Rechne die Wellenlängen in Energien um
    E = constants.h * constants.c / wl * 1e9 / constants.e
    print('Die Abzisse wurde von nm in eV umgerechnet.')

    return E, I


def read_asc_file(datei):
    '''
    Liest eine ascii-Messdatei aus, die von Andor Solis erstellt wurde.

    Parameter:
        datei : String
        Name/Pfad der Messdatei

    Ausgaben:
        E : numpy Array
        Energiewerte des Spektrums

        I : numpy Array
        Intensitätswerte des Spektrums
    '''
    umrechnung = False

    with open(datei) as Messdaten:
        csvReader = csv.reader(Messdaten, delimiter=';')

        E = np.array([])
        I = np.array([])

        for row in csvReader:
            if len(row) != 2:
                # Breche ab
                return E, I
            else:
                Eval, Ival = np.array(row, dtype=np.float)
                # Rechne bei Bedarf von nm in eV um
                if Eval > 100:
                    Eval = constants.h * constants.c / Eval * 1e9 / constants.e
                    umrechnung = True

                # Speichere die Daten
                E = np.append(E, Eval)
                I = np.append(I, Ival)
        if umrechnung:
            print('Die Abzisse wurde von nm in eV umgerechnet.')

        return E, I

def read_file(datei, E_int=(-np.inf, np.inf)):
    '''
    Liest eine Messdatei (entweder eine xlsx-Datei aus dem LabView Programm
    oder eine asc-Datei von Andor Solis) und gibt Energie und Intensität aus.
    '''

    dateiendung = os.path.splitext(datei)[1]

    if dateiendung == '.xlsx':
        E, I = read_excel_file(datei)
    elif dateiendung == '.asc':
        E, I = read_asc_file(datei)
    else:
        raise Exception("Ungueltige Dateiendung (weder .xlsx noch .asc)")

    # Streiche die Wertepaare, deren Energie nicht im Interval E_int liegt
    E_test = [E_int[0] <= ele <= E_int[1] for ele in E]
    E = [E[i] for i in range(len(E)) if E_test[i]]
    I = [I[i] for i in range(len(E)) if E_test[i]]

    return np.array(E), np.array(I)


















