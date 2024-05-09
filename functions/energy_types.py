import numpy as np
from functions import LOGGER

def EnergyFFT(data: np.ndarray, TimeWindow: int,
              Hzini: int = 3, Hzend: int = 45,
              ) -> np.ndarray:
    """
    Calcula la energia de una banda frecuencial definida en el
    siguiente intervalo
    [Hzini, Hzend )

    :parameter data: conjunto de datos para calcular el Waterfall
    :parameter TimeWindow: ventana de solapamiento, como es FFT tiene que
        ser del mismo tamaño que la frecuencia de muestreo. Este código no
        contempla ventanas de FFT menores o mayores
    :parameter Hzini: frecuencia inicial para integrar
    :parameter Hzend: frecuencia final para integrar
    :return: Devuelve la matriz de energía
    """
    LOGGER.debug('[+] Calculating waterfall...')

    NProcessedPoints, NTimeSamples = data.shape
    NTimeWindows = np.floor(NTimeSamples / TimeWindow)

    AcousticEnergyAverage = np.zeros((NProcessedPoints, int(NTimeWindows)))

    for i in range(int(NTimeWindows)):

        data_fft_norm = np.abs(np.fft.fft(data[:, i*TimeWindow:TimeWindow*(i+1)], axis=1))/TimeWindow
        data_fft_norm[:, 0] = 0
        AcousticEnergyAverage[:, i] = np.mean(data_fft_norm[:, Hzini:Hzend+1], axis=1)

    return AcousticEnergyAverage
