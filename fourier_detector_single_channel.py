import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
from functions.hdas_class import HDAS
import seaborn as sns
import os
from scipy.fft import fft, fftfreq

"""
INSERT FILEPATH AND DURATION IN SECONDS BELOW
"""

filepath = "AK_Data/2022_05_05_22*"

def determine_earthquake_threshold(data, sampling_rate, channel_index, freq_range=(1, 10)):
    num_samples = len(data[channel_index])

    channel_fft = fft(data[channel_index])
    fft_freqs = fftfreq(num_samples, d=1/sampling_rate)
    magnitudes = np.abs(channel_fft)

    relevant_indices = np.where((fft_freqs >= freq_range[0]) & (fft_freqs <= freq_range[1]))[0]
    max_magnitude_in_range = np.max(magnitudes[relevant_indices])
    
    print(f"The max magnitude of any frequency in this range in this minute is: {max_magnitude_in_range:.2f}")

def detect_earthquake_fft_single_channel(data, sampling_rate, channel_index, minute_counter, threshold=50000):
    num_samples = len(data[channel_index])
    
    channel_fft = fft(data[channel_index])
    fft_freqs = fftfreq(num_samples, d=1/sampling_rate)
    magnitudes = np.abs(channel_fft)

    significant_indices = np.where((fft_freqs >= 1) & (fft_freqs <= 10) & (magnitudes > threshold))[0]
    
    if len(significant_indices) > 0:
        print("Earthquake may be detected in channel", channel_index)
        plt.figure(figsize=(12, 6))
        plt.plot(fft_freqs[:num_samples//2], magnitudes[:num_samples//2])
        plt.title(f'Frequency Spectrum for Channel {channel_index}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.grid(True)
        plt.savefig(f"output/FFT_plot_minute_{minute_counter}")

    else:
        print(f"No significant seismic activity detected in this channel in minute {minute_counter}")


def denoise_and_detect(data, channel_index):
    strain_max = 5000
    strain_min = -5000

    sampling_rate = 60000 / (10 * 60)

    for z in range(0, 3600, 60):
        sample_start = z * 100
        print("Denoising data...")
        for i in tqdm(range(sample_start, (sample_start + (60 * 100)))):
            if data[channel_index][i] > strain_max or data[channel_index][i] < strain_min:
                data[channel_index][i] = 0

        heatmap_data = data[:, sample_start:(sample_start + (60 * 100))]

        determine_earthquake_threshold(heatmap_data, sampling_rate, channel_index)
        detect_earthquake_fft_single_channel(heatmap_data, sampling_rate, channel_index, int(z/60))

def plot_hdas_from_file(filepath, channel_index):

    bins = np.sort(glob.glob(filepath))

    if len(bins) == 1:
        hdas_data = HDAS(bins[0], load=True)
        denoise_and_detect(hdas_data.Data, channel_index)

    else:
        for i in tqdm(range(len(bins))):
            hdas_data = HDAS(bins[i], load=True)
            if i == 0:
                combined_data = hdas_data.Data
            else:
                combined_data = np.concatenate((combined_data, hdas_data.Data), axis=1)
        
        denoise_and_detect(combined_data, channel_index)

plot_hdas_from_file(filepath, 1500)