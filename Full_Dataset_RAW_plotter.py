import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, butter, filtfilt, decimate
import glob
from tqdm import tqdm
from functions.hdas_class import HDAS
from functions.laser_denoising import *
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
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
    freq_range = (30, 45)  # Frequency range for band-pass filter
    channel_range = (0, 1000)  # optional
    decimation_factor = None

    def bandpass_filter(data, sampling_rate, freq_range, order=1):

        nyquist = 0.5 * sampling_rate
        low = freq_range[0] / nyquist
        high = freq_range[1] / nyquist
        b, a = butter(order, [low, high], btype='band')
        filtered_data = filtfilt(b, a, data)
        return filtered_data

    def plot_band_power_heatmap_stft(data, sampling_rate, duration_seconds, starting_seconds, freq_range, channel_range=None):
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

        filtered_data = []

        print("Filtering Data in specified freq. range...")
        for channel_index in tqdm(range(num_channels)):
            filtered_row = bandpass_filter(heatmap_data[channel_index],sampling_rate,freq_range)
            filtered_data.append(filtered_row)

        # filtered_data = bandpass_filter(heatmap_data, sampling_rate, freq_range)

        if duration_seconds > 60:
            time_label = 'Time (minutes)'
            ticks_array = np.arange(0, total_samples + 1, 3000)
            labels_array = np.arange(0, (duration_seconds + 1 )/ 60, 0.5)

        else:
            time_label = 'Time (seconds)'
            ticks_array = np.arange(0, total_samples + 1, 100)
            labels_array = np.arange(0, duration_seconds + 1, 1)

        plt.figure(figsize=(12, 8))

        vmin = np.percentile(filtered_data, 40)
        vmax = np.percentile(filtered_data, 90)

        cmap = LinearSegmentedColormap.from_list("TealOrange", ["teal", "orange"])

        # sns.heatmap(filtered_data, cmap='viridis', cbar=True, xticklabels=labels_array, 
        #             cbar_kws={'label': 'Strain Data Value Post Bandpass Filter'}, vmin=vmin, vmax=vmax)

        plt.imshow(filtered_data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        plt.colorbar()    

        plt.xlabel(time_label)
        plt.ylabel('Channel')
        plt.title(f'Strain Values after {freq_range[0]}-{freq_range[1]}Hz Bandpass Filter')
        plt.yticks(ticks=np.arange(0, num_channels + 1, 500), labels=np.arange(channel_range[0], channel_range[1] + 1, 500))
        plt.xticks(ticks=ticks_array, labels=labels_array)
        plt.gca().invert_yaxis()

        output_dir = "RAW_BP_2022_05_15_05_30-45Hz"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, f"{filename}.png"))
        plt.close()
        print(f"{filename}.png...")

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

<<<<<<< HEAD
    plot_hdas_from_file(filepath, duration_seconds, starting_seconds, channel_range)
=======
    plot_hdas_from_file(filepath, duration_seconds, starting_seconds, channel_range)
>>>>>>> 5c98c2c4665861561c78c8a89c8636c6ddecbb16
