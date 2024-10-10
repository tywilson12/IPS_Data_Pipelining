import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, butter, filtfilt
import glob
from tqdm import tqdm
from functions.hdas_class import HDAS
from functions.laser_denoising import *
import seaborn as sns
import os

"""
INSERT FILEPATH AND DURATION IN SECONDS BELOW
"""

folder_path = "AK_Data"

all_files = [filename for filename in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, filename)) and 
             filename.startswith("2022_05_15_05")]

for filename in tqdm(all_files):
 
    filepath = os.path.join(folder_path, filename)
    duration_seconds = 600
    starting_seconds = 0
    freq_range = (10, 30)  # Frequency range for band-pass filter
    channel_index = 500

    def bandpass_filter(data, sampling_rate, freq_range, order=5):
        nyquist = 0.5 * sampling_rate
        low = freq_range[0] / nyquist
        high = freq_range[1] / nyquist
        b, a = butter(order, [low, high], btype='band')
        filtered_data = filtfilt(b, a, data)
        return filtered_data

    def plot_single_channel_spectrogram(data, sampling_rate, duration_seconds, starting_seconds, channel_index, freq_range):
        sample_start = int(starting_seconds * sampling_rate)
        total_samples = int(sampling_rate * duration_seconds)
        
        data_segment = data[channel_index, sample_start:sample_start + total_samples]

        filtered_data = bandpass_filter(data_segment, sampling_rate, freq_range)

        plt.figure(figsize=(12, 8))

        vmin = np.percentile(filtered_data, 0)
        vmax = np.percentile(filtered_data, 25)
        
        Pxx, freqs, bins, im = plt.specgram(filtered_data, NFFT=100, Fs=sampling_rate, noverlap=50, cmap='viridis', vmax=vmax, vmin=vmin)
        
        plt.ylim(freq_range)
        plt.colorbar(im, label='Power/Frequency (dB/Hz)')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency (Hz)')
        plt.title(f'Spectrogram for Channel {channel_index} in Frequency Range {freq_range}')
        
        output_dir = "specgram_2022_05_15_05_10-30Hz"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, f"{filename}.png"))
        plt.close()
        print(f"Plot saved as {filename}.png...")

    def plot_hdas_from_file(filepath, duration_seconds, starting_seconds, channel_index, freq_range):
        bins = np.sort(glob.glob(filepath))

        if len(bins) == 1:
            hdas_data = HDAS(bins[0], load=True)

            hdas_data.Data, HDAS_LaserRef, aditional_out = laserDenoisingRefFiberv2(
                data=hdas_data,
                FiberRefStop=hdas_data.FiberRefStop,
                FiberRefStart=hdas_data.FiberRefStart,
                RawData=hdas_data.RawMode
            )

            plot_single_channel_spectrogram(hdas_data.Data, 60000 / (10 * 60), duration_seconds, starting_seconds, channel_index, freq_range)

        else:
            combined_data = None
            for i in tqdm(range(len(bins))):
                hdas_data = HDAS(bins[i], load=True)

                hdas_data.Data, HDAS_LaserRef, aditional_out = laserDenoisingRefFiberv2(
                    data=hdas_data,
                    FiberRefStop=hdas_data.FiberRefStop,
                    FiberRefStart=hdas_data.FiberRefStart,
                    RawData=hdas_data.RawMode
                )

                if i == 0:
                    combined_data = hdas_data.Data
                else:
                    combined_data = np.concatenate((combined_data, hdas_data.Data), axis=1)
            
            plot_single_channel_spectrogram(combined_data, 60000 / (10 * 60), duration_seconds, starting_seconds, channel_index, freq_range)

    try:
        plot_hdas_from_file(filepath, duration_seconds, starting_seconds, channel_index, freq_range)
    except Exception as e:
        print(f"An error occured when trying to process {filename}: {e}")
        continue