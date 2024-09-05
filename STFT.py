import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
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
starting_seconds = 840
freq_range = (0, 3)

def calculate_band_power_stft(data, sampling_rate, freq_range, nperseg=100): # May need to modify nperseg based on segment length

    freqs, times, Zxx = stft(data, fs=sampling_rate, nperseg=nperseg, noverlap=nperseg//2)
    
    freq_indices = np.where((freqs >= freq_range[0]) & (freqs <= freq_range[1]))[0]
    
    band_power = np.sum(np.abs(Zxx[freq_indices, :])**2, axis=0)
    
    return band_power, times

def plot_band_power_heatmap_stft(data, sampling_rate, duration_seconds, starting_seconds, freq_range):

    segment_length = 5  # Length of each segment in seconds
    segment_samples = int(segment_length * sampling_rate)
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

    power_matrix = []

    print("Calculating power in specified freq. range...")
    for channel_index in tqdm(range(num_channels)):
        band_power, times = calculate_band_power_stft(heatmap_data[channel_index, :], sampling_rate, freq_range, nperseg=segment_samples)
        power_matrix.append(band_power)

    power_matrix = np.array(power_matrix)
    
    if duration_seconds > 60:
        times = times / 60.0
        time_label = 'Time (minutes)'
    else:
        time_label = 'Time (seconds)'

    plt.figure(figsize=(12, 8))
    vmin = np.min(power_matrix)
    vmax = np.max(power_matrix)

    vmin = 1000000.0

    sns.heatmap(power_matrix, cmap='coolwarm', cbar=True, xticklabels=int(sampling_rate * segment_length),
                cbar_kws={'label': 'Power (a.u.)'}, vmin=vmin, vmax=vmax)

    plt.xlabel(time_label)
    plt.ylabel('Channel')
    plt.title(f'Power in {freq_range[0]}-{freq_range[1]} Hz Band (STFT)')
    plt.xticks(ticks=np.arange(0, len(times), len(times)//10), labels=np.round(times[np.arange(0, len(times), len(times)//10)], 2))
    plt.yticks(ticks=np.arange(0, num_channels, 500), labels=np.arange(0, num_channels, 500))
    plt.gca().invert_yaxis()

    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, "band_power_heatmap_stft.png"))
    plt.close()
    print("Plot Saved...")

def plot_hdas_from_file(filepath, duration_seconds, starting_seconds):

    bins = np.sort(glob.glob(filepath))

    if len(bins) == 1:
        hdas_data = HDAS(bins[0], load=True)
        plot_band_power_heatmap_stft(hdas_data.Data, 60000 / (10 * 60), duration_seconds, starting_seconds, freq_range)

    else:
        for i in tqdm(range(len(bins))):
            hdas_data = HDAS(bins[i], load=True)
            if i == 0:
                combined_data = hdas_data.Data
            else:
                combined_data = np.concatenate((combined_data, hdas_data.Data), axis=1)
        
        plot_band_power_heatmap_stft(combined_data, 60000 / (10 * 60), duration_seconds, starting_seconds, freq_range)

plot_hdas_from_file(filepath, duration_seconds, starting_seconds)