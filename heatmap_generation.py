import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
from functions.hdas_class import HDAS
import seaborn as sns

""" 
INSERT FILENAME BELOW
"""

filename = "AK_Data/2022_05_19_02h29m43s_HDAS_2DRawData_Strain.bin"
duration = 10

def plot_hdas_data_heatmap(data, duration_seconds):
    strain_max = 5000
    strain_min = -5000
    high_strain_threshold = 1000
    low_strain_threshold = -1000

    sampling_rate = 60000 / (10 * 60)
    total_samples = int(sampling_rate * duration_seconds)

    if total_samples > len(data[0]):
        raise ValueError(f"The maximum available data length is {len(data[0]) / sampling_rate} seconds. Reduce the duration.")

    num_channels = len(data)

    print("Denoising data...")
    for i in tqdm(range(total_samples)):
        for j in range(num_channels):
            if data[j][i] > strain_max or data[j][i] < strain_min:
                data[j][i] = 0

    heatmap_data = data[:, :total_samples]
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

    print("Saving heatmap...")
    if duration_seconds <= 60:
        plt.savefig(f"heatmap_plot_{duration_seconds}_seconds.png")
    else:
        plt.savefig(f"heatmap_plot_{duration_minutes}_minutes.png")
    print("Heatmap saved...")

def plot_hdas_from_file(file_path, duration):

    bins = np.sort(glob.glob(file_path))

    if len(bins) == 1:
        hdas_data = HDAS(bins[0], load=True)
        plot_hdas_data_heatmap(hdas_data.Data, duration)

    else:
        for i in tqdm(range(len(bins))):
            hdas_data = HDAS(bins[i], load=True)
            if i == 0:
                combined_data = hdas_data.Data
            else:
                combined_data = np.concatenate((combined_data, hdas_data.Data), axis=1)
        
        plot_hdas_data_heatmap(combined_data, duration)

plot_hdas_from_file(filename, duration)
