import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, butter, filtfilt, decimate
import glob
from tqdm import tqdm
from functions.hdas_class import HDAS
from functions.laser_denoising import *
import seaborn as sns
from scipy.ndimage import uniform_filter1d
import os

"""
INSERT FILEPATH AND DURATION IN SECONDS BELOW
"""

filepath = "AK_Data/2022_05_05_20*"
duration_seconds = 180
starting_seconds = 780
freq_range = (0.5, 10)  # Frequency range for band-pass filter
channel_range = None  # optional
decimation_factor = None

def bandpass_filter(data, sampling_rate, freq_range, order=5):

    nyquist = 0.5 * sampling_rate
    low = freq_range[0] / nyquist
    high = freq_range[1] / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def calculate_band_power_stft(data, sampling_rate, freq_range, nperseg=100): 

    if decimation_factor != None:
        data = decimate(data, decimation_factor, axis=-1, zero_phase=True).astype(np.float32)
        sampling_rate = sampling_rate/decimation_factor

    filtered_data = bandpass_filter(data, sampling_rate, freq_range)
    
    freqs, times, Zxx = stft(filtered_data, fs=sampling_rate, nperseg=nperseg, noverlap=nperseg//2)
    
    freq_indices = np.where((freqs >= freq_range[0]) & (freqs <= freq_range[1]))[0]
    
    band_power = np.sum(np.abs(Zxx[freq_indices, :]), axis=0)
    
    return band_power, times

def plot_band_power_heatmap_stft(data, sampling_rate, duration_seconds, starting_seconds, freq_range, channel_range=None):
    segment_length = 1  # Length of each segment in seconds
    segment_samples = int(segment_length * sampling_rate)
    sample_start = int(starting_seconds * sampling_rate)
    total_samples = int(sampling_rate * duration_seconds)
    
    if total_samples > data.shape[1] - sample_start:
        raise ValueError(f"The maximum available data length is {(data.shape[1] - sample_start) / sampling_rate} seconds. Reduce the duration.")
    
    if channel_range == None:
        num_channels = data.shape[0]
        heatmap_data = data[:, sample_start:(sample_start + total_samples)]
    else:
        num_channels = channel_range[1] - channel_range[0]
        heatmap_data = data[channel_range[0]:channel_range[1], sample_start:(sample_start + total_samples)]

    power_matrix = []

    print("Calculating power in specified freq. range...")
    for channel_index in tqdm(range(num_channels)):
        band_power, times = calculate_band_power_stft(heatmap_data[channel_index], sampling_rate, freq_range, nperseg=segment_samples)
        power_matrix.append(band_power)

    power_matrix = np.array(power_matrix)

    if duration_seconds > 60:
        times = times / 60.0
        time_label = 'Time (minutes)'
    else:
        time_label = 'Time (seconds)'

    plt.figure(figsize=(12, 8))
    vmin = np.percentile(power_matrix, 0)
    vmax = np.percentile(power_matrix, 80)

    sns.heatmap(power_matrix, cmap='Purples', cbar=True, xticklabels=int(sampling_rate * segment_length),
                cbar_kws={'label': 'Power in Freq. Band'}, vmin=vmin, vmax=vmax)

    plt.xlabel(time_label)
    plt.ylabel('Channel')
    plt.title(f'Power in {freq_range[0]}-{freq_range[1]} Hz Band (STFT)')
    plt.xticks(ticks=np.arange(0, len(times), len(times)//(duration_seconds//60)), labels=np.round(times[np.arange(0, len(times), len(times)//(duration_seconds//60))], 2))
    plt.yticks(ticks=np.arange(0, num_channels, 500), labels=np.arange(0, num_channels, 500))
    plt.gca().invert_yaxis()

    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, "band_power_heatmap_stft.png"))
    plt.close()
    print("Plot Saved as band_power_heatmap_stft.png...")

def plot_hdas_from_file(filepath, duration_seconds, starting_seconds, channel_range=None):

    bins = np.sort(glob.glob(filepath))

    if len(bins) == 1:

        hdas_data = HDAS(bins[0], load=True)

        [hdas_data.Data, HDAS_LaserRef, aditional_out] = laserDenoisingRefFiberv2(data=hdas_data,
                                                                              FiberRefStop=hdas_data.FiberRefStop,
                                                                              FiberRefStart=hdas_data.FiberRefStart,
                                                                              RawData=hdas_data.RawMode
                                                                              )

        plot_band_power_heatmap_stft(hdas_data.Data, 60000 / (10 * 60), duration_seconds, starting_seconds, freq_range, channel_range)

    else:
        for i in tqdm(range(len(bins))):

            hdas_data = HDAS(bins[i], load=True)

            [hdas_data.Data, HDAS_LaserRef, aditional_out] = laserDenoisingRefFiberv2(data=hdas_data,
                                                                              FiberRefStop=hdas_data.FiberRefStop,
                                                                              FiberRefStart=hdas_data.FiberRefStart,
                                                                              RawData=hdas_data.RawMode
                                                                              )

            if i == 0:
                combined_data = hdas_data.Data
            else:
                combined_data = np.concatenate((combined_data, hdas_data.Data), axis=1)
        
        plot_band_power_heatmap_stft(combined_data, 60000 / (10 * 60), duration_seconds, starting_seconds, freq_range, channel_range)

plot_hdas_from_file(filepath, duration_seconds, starting_seconds, channel_range)