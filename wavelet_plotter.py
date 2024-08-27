import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
import glob
from tqdm import tqdm
from functions.hdas_class import HDAS
import os

"""
INSERT FILEPATH AND DURATION IN SECONDS BELOW
"""

filepath = "AK_Data/2022_05_05_20*"
duration_seconds = 120
starting_seconds = 240

def haar_wavelet_transform(data):
    haar_low_pass = np.array([1, 1]) / np.sqrt(2)
    haar_high_pass = np.array([1, -1]) / np.sqrt(2)

    approximation = convolve(data, haar_low_pass, mode='valid')[::2]
    detail = convolve(data, haar_high_pass, mode='valid')[::2]

    return approximation, detail

def plot_hdas_data_wavelet(data, duration_seconds, starting_seconds):
    strain_max = 5000
    strain_min = -5000

    sampling_rate = 60000 / (10 * 60)
    total_samples = int(sampling_rate * duration_seconds)
    sample_start = starting_seconds * 100

    if total_samples > len(data[0]):
        raise ValueError(f"The maximum available data length is {len(data[0]) / sampling_rate} seconds. Reduce the duration.")

    num_channels = len(data)

    print("Denoising data...")
    for i in tqdm(range(sample_start, (sample_start + (duration_seconds * 100)))):
        for j in range(len(data)):
            if data[j][i] > strain_max or data[j][i] < strain_min:
                data[j][i] = 0

    wavelet_data = []
    for channel in range(num_channels):
        _, detail = haar_wavelet_transform(data[channel, sample_start:(sample_start + (duration_seconds * 100))])
        wavelet_data.append(detail)

    wavelet_data = np.array(wavelet_data)

    plt.figure(figsize=(20, 10))

    mean = np.mean(wavelet_data)
    std = np.std(wavelet_data)
    vmin = mean - 1 * std  # Two standard deviations below the mean
    vmax = mean + 1 * std  # Two standard deviations above the mean
    plt.imshow(wavelet_data, aspect='auto', cmap='coolwarm', extent=[0, duration_seconds, 0, num_channels], vmin=vmin, vmax=vmax)
    plt.colorbar(label='Wavelet Detail Coefficient')
    plt.gca().invert_yaxis()

    if duration_seconds <= 60:
        plt.xlabel('Time (seconds)')
        plt.title(f'Wavelet Detail Coefficients of Channels Over {duration_seconds} Seconds')
    else:
        duration_minutes = duration_seconds / 60
        plt.xlabel('Time (minutes)')
        plt.title(f'Wavelet Detail Coefficients of Channels Over {duration_minutes} Minutes')
        minute_ticks = np.arange(0, duration_minutes + 1, 1)
        second_ticks = minute_ticks * 60
        plt.xticks(second_ticks, minute_ticks)

    plt.ylabel('Channel')
    plt.gca().invert_yaxis()

    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving wavelet plot...")
    if duration_seconds <= 60:
        output_filepath = os.path.join(output_dir, f"wavelet_plot_{duration_seconds}_seconds.png")
        plt.savefig(output_filepath)
    else:
        output_filepath = os.path.join(output_dir, f"wavelet_plot_{duration_minutes}_minutes.png")
        plt.savefig(output_filepath)
    print("Wavelet plot saved...")

def plot_hdas_from_file(filepath, duration_seconds, starting_seconds):

    bins = np.sort(glob.glob(filepath))

    if len(bins) == 1:
        hdas_data = HDAS(bins[0], load=True)
        plot_hdas_data_wavelet(hdas_data.Data, duration_seconds, starting_seconds)

    else:
        for i in tqdm(range(len(bins))):
            hdas_data = HDAS(bins[i], load=True)
            if i == 0:
                combined_data = hdas_data.Data
            else:
                combined_data = np.concatenate((combined_data, hdas_data.Data), axis=1)
        
        plot_hdas_data_wavelet(combined_data, duration_seconds, starting_seconds)

plot_hdas_from_file(filepath, duration_seconds, starting_seconds)