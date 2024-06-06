import numpy as np
import glob
from scipy import signal
import os
import h5py
import matplotlib.pyplot as plt
import csv
import tqdm

from functions import LOGGER
from functions.energy_types import EnergyFFT
from functions.plotting import plot_waterfall
from functions.plotting import plot_laserRef
from functions.plotting import plot_strain
from functions.hdas_class import HDAS
from functions.laser_denoising import *
from functions import LOGGER

filename = "AK_Data/2022_05_19_07h*"

bins=np.sort(glob.glob(filename))

filename = filename.replace('/', '_')

for i in tqdm.tqdm(range(len(bins))):

    hdas_data = HDAS(bins[i], load=True)

    [hdas_data.Data, HDAS_LaserRef, additional_out] = laserDenoisingRefFiberv2(data=hdas_data,
                                                                                FiberRefStop=hdas_data.FiberRefStop,
                                                                                FiberRefStart=hdas_data.FiberRefStart,
                                                                                RawData=hdas_data.RawMode
                                                                                )

    print(len(hdas_data.Data), len(hdas_data.Data[0]))

    if i == 0:
        combined_data = hdas_data.Data
    else:
        combined_data = np.concatenate((combined_data, hdas_data.Data), axis=1)

x_axis_values1 = []
x_axis_values2 = []

for i in range(0,len(bins) * 2 + 1):
    x_axis_values1.append(i * 30000)
    x_axis_values2.append(i * 5)

plt.figure(dpi=300)
plt.plot(combined_data[1500],lw=0.1)
plt.title('Laser denoised data, ch. 1500')
plt.xlabel('min')
plt.ylabel('Strain')
plt.xticks(x_axis_values1,x_axis_values2)
plt.savefig(f"periodogram_{filename}.png")

# plt.figure(dpi=300)
# f, Pxx_den = signal.periodogram(signal.detrend(hdas_data.Data[1500,:]),100)
# plt.semilogx(f, 10*np.log10(Pxx_den),lw=1)
# plt.ylim(-50,50)
# plt.xlabel('frequency [Hz]')
# plt.ylabel('PSD $strain**2/Hz (dB) $')


# np.savetxt("output_data.csv", hdas_data.Data[0], delimiter=",")
