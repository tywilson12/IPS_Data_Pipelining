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

def plot_hdas_data_heatmap(data, duration_seconds, starting_seconds):
    strain_max = 5000
    strain_min = -5000
    high_strain_threshold = 1000
    low_strain_threshold = -1000

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

    heatmap_data = data[:, sample_start:(sample_start + (duration_seconds * 100))]
    # heatmap_data[np.abs(heatmap_data) < high_strain_threshold] = 0

    time_vector = np.linspace(0, duration_seconds, total_samples)

    plt.figure(figsize=(20, 10))
    
    ax = sns.heatmap(heatmap_data, cmap='coolwarm', cbar=True, xticklabels=int(sampling_rate), yticklabels=500,
                     cbar_kws={'label': 'Strain Value'}, vmin=-2000, vmax=2000)
    
    if duration_seconds <= 60:
        second_intervals = np.arange(0, duration_seconds + 1, 1)
        tick_positions = np.linspace(0, total_samples, len(second_intervals))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(second_intervals)
        plt.xlabel('Time (seconds)')
        plt.title(f'Heatmap of Channels Over {duration_seconds} Seconds')
    else:
        duration_minutes = duration_seconds / 60
        minute_intervals = np.arange(0, duration_minutes + 1, 1)
        tick_positions = np.linspace(0, total_samples, len(minute_intervals))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(minute_intervals)
        plt.xlabel('Time (minutes)')
        plt.title(f'Heatmap of Channels Over {duration_minutes} Minutes')

    ax.invert_yaxis()
    plt.ylabel('Channel')

    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving heatmap...")
    if duration_seconds <= 60:
        output_filepath = os.path.join(output_dir, f"heatmap_plot_{duration_seconds}_seconds.png")
        plt.savefig(output_filepath)
    else:
        output_filepath = os.path.join(output_dir, f"heatmap_plot_{duration_minutes}_minutes.png")
        plt.savefig(output_filepath)
    print("Heatmap saved...")

def plot_hdas_from_file(filepath, duration_seconds, starting_seconds):

    bins = np.sort(glob.glob(filepath))

    if len(bins) == 1:
        hdas_data = HDAS(bins[0], load=True)
        plot_hdas_data_heatmap(hdas_data.Data, duration_seconds)

    else:
        for i in tqdm(range(len(bins))):
            hdas_data = HDAS(bins[i], load=True)
            if i == 0:
                combined_data = hdas_data.Data
            else:
                combined_data = np.concatenate((combined_data, hdas_data.Data), axis=1)
        
        plot_hdas_data_heatmap(combined_data, duration_seconds, starting_seconds)

plot_hdas_from_file(filepath, duration_seconds, starting_seconds)
