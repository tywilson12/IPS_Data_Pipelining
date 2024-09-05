import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
from functions.hdas_class import HDAS
import seaborn as sns
import os

"""
INSERT FILEPATH AND DURATION IN SECONDS BELOW
"""

filepath = "AK_Data/2022_05_05_20*"
duration_seconds = 60
starting_seconds = 780
freq_range = (-15, 15)

def calculate_fk_transform(data, sampling_rate):
    fk_transform = np.fft.fftshift(np.fft.fft2(data))

    num_samples = data.shape[1]
    frequencies = np.fft.fftfreq(num_samples, d=1/sampling_rate)
    frequencies_shifted = np.fft.fftshift(frequencies)

    num_channels = data.shape[0]
    wavenumbers = np.fft.fftfreq(num_channels)
    wavenumbers_shifted = np.fft.fftshift(wavenumbers)

    return fk_transform, frequencies_shifted, wavenumbers_shifted

def plot_fk_heatmap(data, sampling_rate, duration_seconds, starting_seconds, freq_range=(-15, 15)):

    sample_start = int(starting_seconds * sampling_rate)
    total_samples = int(sampling_rate * duration_seconds)
    
    if total_samples > data.shape[1] - sample_start:
        raise ValueError(f"The maximum available data length is {(data.shape[1] - sample_start) / sampling_rate} seconds. Reduce the duration.")
    
    num_channels = data.shape[0]
    
    strain_max = 5000
    strain_min = -5000

    print("Denoising Data...")
    for i in tqdm(range(sample_start, (sample_start + total_samples))):
        for j in range(len(data)):
            if data[j][i] > strain_max or data[j][i] < strain_min:
                data[j][i] = 0

    heatmap_data = data[:, sample_start:(sample_start + total_samples)]

    print("Calculating FK transform...")
    fk_transform, frequencies_shifted, wavenumbers_shifted = calculate_fk_transform(heatmap_data, sampling_rate)

    freq_mask = (frequencies_shifted >= freq_range[0]) & (frequencies_shifted <= freq_range[1])

    fk_transform_filtered = fk_transform[:, freq_mask]

    fk_transform_log = np.log(np.abs(fk_transform_filtered) + 1) #+1 to avoid log 0

    plt.figure(figsize=(12, 8))
    
    vmin = np.percentile(fk_transform_log, 5)
    vmax = np.percentile(fk_transform_log, 95)
    
    plt.imshow(fk_transform_log, extent=[frequencies_shifted[freq_mask][0], frequencies_shifted[freq_mask][-1], wavenumbers_shifted[0], wavenumbers_shifted[-1]], 
               aspect='auto', cmap='inferno', vmin=vmin, vmax=vmax)
    
    plt.colorbar(label='Log Amplitude')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Wavenumber (10m Spacing)')
    plt.title(f'FK Spectrum (Log Scale, {freq_range[0]}-{freq_range[1]} Hz)')
    plt.show()

    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, "fk_spectrum_log_scale_filtered.png"))
    plt.close()
    print("Plot Saved...")

def plot_hdas_from_file(filepath, duration_seconds, starting_seconds, freq_range=(1, 15)):

    bins = np.sort(glob.glob(filepath))

    if len(bins) == 1:
        hdas_data = HDAS(bins[0], load=True)
        plot_fk_heatmap(hdas_data.Data, 60000 / (10 * 60), duration_seconds, starting_seconds, freq_range=freq_range)

    else:
        for i in tqdm(range(len(bins))):
            hdas_data = HDAS(bins[i], load=True)
            if i == 0:
                combined_data = hdas_data.Data
            else:
                combined_data = np.concatenate((combined_data, hdas_data.Data), axis=1)
        
        plot_fk_heatmap(combined_data, 60000 / (10 * 60), duration_seconds, starting_seconds, freq_range=freq_range)

plot_hdas_from_file(filepath, duration_seconds, starting_seconds, freq_range)