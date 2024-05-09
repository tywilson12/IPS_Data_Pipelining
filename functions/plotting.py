import matplotlib.pyplot as plt
import numpy as np
from functions import LOGGER


def plot_colors():
    return np.array(['k', 'b', 'g', 'r', 'c', 'm', 'y', 'tab:orange', 'tab:gray', 'tab:pink'])


def plot_strain(hdas_object, Array_to_Plot_indexes):
    colors = plot_colors()

    if hdas_object.Headers[101] == 5:
        Array_to_Plot_indexes = np.arange(0, 10)
    if hdas_object.Headers[101] == 6:
        Array_to_Plot_indexes = np.arange(0, 10)
    if hdas_object.Headers[101] == 7:
        Array_to_Plot_indexes = [1]
    if hdas_object.Headers[101] == 8:
        Array_to_Plot_indexes = [1]

    Array_to_Plot_meters = (Array_to_Plot_indexes) * hdas_object.Spatial_Sampling_Meters + \
                           hdas_object.Fiber_Position_Offset

    time_xx = np.arange((hdas_object.TimeStart + 1) / hdas_object.Trigger_Frequency,
                        (hdas_object.TimeStop + 1) / hdas_object.Trigger_Frequency,
                        1 / hdas_object.Trigger_Frequency)

    fig1, axs = plt.subplots()
    axs.plot(time_xx, hdas_object.Data[int(hdas_object.FiberRefStart - 1), hdas_object.TimeStart:hdas_object.TimeStop],
             label='Fiber Reference', color=colors[0])
    for i in np.arange(0, Array_to_Plot_meters.shape[0]):
        axs.plot(time_xx, hdas_object.Data[Array_to_Plot_indexes[i] - 1, hdas_object.TimeStart:hdas_object.TimeStop],
                 label='Meter ' + str(Array_to_Plot_meters[i]),
                 color=colors[i + 1])
    axs.set_xlim([time_xx[0], time_xx[-1]])
    axs.legend(loc='upper right')
    axs.set_xlabel('Time (s)')
    axs.set_ylabel(' STrain Variation (nStrain)')
    axs.set_title('Strain along Time at Positions MultiPoint')
    fig1.tight_layout()
    print("works?")


def plot_laserRef(laserRef, RawData):
    if RawData:
        LOGGER.info('[+] Plotting laser ref ... ')
        fig2, axs = plt.subplots()
        axs.plot(laserRef, label='Laser Ref')
        axs.plot(np.diff(laserRef), label='diff Laser Ref')
        axs.set_title("HDAS Laser Ref")
        axs.set_ylabel("Strain Variation (nStrain)")
        axs.set_xlabel("Time (samples)")
        axs.set_xlim([0, laserRef.shape[0]])
        axs.legend(loc='upper right')
        fig2.tight_layout()


def plot_waterfall(nombre_fichero: str, matrix_waterfall: np.ndarray,
                   Hzini: int = 4, Hzend: int = 45,
                   SpatialSamplingMeters: float = 10, FiberPositionOffset: float = 1,
                   TimeStart: int = 0, Trigger_Frequency: float = 250,
                   TimeWindow_seconds: int = 10, N_TimeWindows: int = 10):

    LOGGER.info('[+] Plotting waterfall ... ')
    fig3, axs = plt.subplots()

    time_xx = np.arange((TimeStart + 1) / Trigger_Frequency,
                        (TimeStart + 1) / Trigger_Frequency + (N_TimeWindows) * TimeWindow_seconds,
                        TimeWindow_seconds)

    maxPx = (matrix_waterfall.shape[0] - 1) * SpatialSamplingMeters + FiberPositionOffset

    lims = [time_xx[0],time_xx[-1],FiberPositionOffset,maxPx]
    dx = float(np.diff(lims[:2])/(matrix_waterfall.shape[1]-1))
    dy = float(-np.diff(lims[2:]) / (matrix_waterfall.shape[0]-1))
    extent = [lims[0] - dx / 2, lims[1] + dx / 2, lims[2] + dy / 2, lims[3] - dy / 2]
    colorbar = axs.imshow(matrix_waterfall, origin='lower',
                          extent=extent, aspect='auto', interpolation='none')

    cbar = plt.colorbar(colorbar)
    axs.set_xlabel('Time (s)')
    axs.set_ylabel('Distance (m)')
    axs.set_title('Acustic Energy (' + str(Hzini) + ' - ' + str(Hzend) +
                  'Hz, Average Energy) VS Time VS Fiber Position')
    fig3.tight_layout()


def plot_HDASRef(HDAS_LaserRef):
    fig, axs = plt.subplots()
    axs.plot(HDAS_LaserRef)
