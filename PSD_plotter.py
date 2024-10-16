import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import glob
from tqdm import tqdm
from functions.hdas_class import HDAS
import seaborn as sns
import os

"""
INSERT FILEPATH AND DURATION IN SECONDS BELOW
"""

filepath = "AK_Data/2022_05_05_20*"
duration_seconds = 600
starting_seconds = 600
freq_range = (1, 5)  # Frequency range of interest (in Hz)

def calculate_band_power(data, sampling_rate, freq_range, nperseg=100):

    freqs, psd = welch(data, fs=sampling_rate, nperseg=nperseg, noverlap=nperseg//2, return_onesided=True)
    
    freq_indices = np.where((freqs >= freq_range[0]) & (freqs <= freq_range[1]))[0]
    
    band_power = np.sum(psd[freq_indices])
    
    return band_power

def plot_band_power_heatmap(data, sampling_rate, duration_seconds, starting_seconds, freq_range):

    segment_length = 5  # Length of each segment in seconds
    segment_samples = int(segment_length * sampling_rate)
    sample_start = int(starting_seconds * sampling_rate)
    total_samples = int(sampling_rate * duration_seconds)
    
    if total_samples > data.shape[1] - sample_start:
        raise ValueError(f"The maximum available data length is {(data.shape[1] - sample_start) / sampling_rate} seconds. Reduce the duration.")
    
    num_channels = data.shape[0]
    num_segments = total_samples // segment_samples

    strain_max = 5000
    strain_min = -5000

    print("Denoising Data...")
    for i in tqdm(range(sample_start, (sample_start + total_samples))):
        for j in range(len(data)):
            if data[j][i] > strain_max or data[j][i] < strain_min:
                data[j][i] = 0

    heatmap_data = data[:, sample_start:(sample_start + total_samples)]

    power_matrix = np.zeros((num_channels, num_segments))

    print("Calculating power in specified freq. range...")
    for channel_index in tqdm(range(num_channels)):
        for seg_index in range(num_segments):
            segment_data = heatmap_data[channel_index, seg_index * segment_samples : (seg_index + 1) * segment_samples]
            band_power = calculate_band_power(segment_data, sampling_rate, freq_range)
            power_matrix[channel_index, seg_index] = band_power
    
    time_vector = np.arange(starting_seconds, starting_seconds + segment_length * num_segments, segment_length)

    if duration_seconds > 60:
        time_vector = time_vector / 60.0
        time_label = 'Time (minutes)'

        full_minutes = np.arange(np.floor(time_vector[0]), np.ceil(time_vector[-1]) + 1, 1)
        tick_positions = np.interp(full_minutes, time_vector, np.arange(num_segments))
    else:
        time_label = 'Time (seconds)'
        full_minutes = time_vector
        tick_positions = np.arange(num_segments)
    
    plt.figure(figsize=(12, 8))
    vmin = np.min(power_matrix)
    vmax = np.max(power_matrix)

    sns.heatmap(power_matrix, cmap='coolwarm', cbar=True, xticklabels=int(sampling_rate * segment_length),
                cbar_kws={'label': 'Power (a.u.)'}, vmin=vmin, vmax=vmax)

    plt.xlabel(time_label)
    plt.ylabel('Channel')
    plt.title(f'Power in {freq_range[0]}-{freq_range[1]} Hz Band (PSD)')

    plt.xticks(ticks=tick_positions, labels=np.round(full_minutes, 2))
    
    plt.yticks(ticks=np.arange(0, num_channels, 500), labels=np.arange(0, num_channels, 500))
    plt.gca().invert_yaxis()

    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, f"band_power_heatmap_PSD.png"))
    plt.close()
    print("Plot Saved...")

def plot_hdas_from_file(filepath, duration_seconds, starting_seconds):

    bins = np.sort(glob.glob(filepath))

    if len(bins) == 1:
        hdas_data = HDAS(bins[0], load=True)
        plot_band_power_heatmap(hdas_data.Data, 60000 / (10 * 60), duration_seconds, starting_seconds, freq_range)

    else:
        for i in tqdm(range(len(bins))):
            hdas_data = HDAS(bins[i], load=True)
            if i == 0:
                combined_data = hdas_data.Data
            else:
                combined_data = np.concatenate((combined_data, hdas_data.Data), axis=1)
        
        plot_band_power_heatmap(combined_data, 60000 / (10 * 60), duration_seconds, starting_seconds, freq_range)

plot_hdas_from_file(filepath, duration_seconds, starting_seconds)