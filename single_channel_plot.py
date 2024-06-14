import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
from functions.hdas_class import HDAS

def plot_hdas_data(data, channel=1500, duration_seconds=600):
    max_strain = 5000
    min_strain = -5000
    
    sample_rate = 60000 / (10 * 60)
    total_samples = int(sample_rate * duration_seconds)
    
    num_samples = len(data[channel])
    
    if total_samples > num_samples:
        raise ValueError(f"The maximum available data length is {num_samples / sample_rate} seconds. Reduce the duration.")

    for i in range(total_samples):
        if data[channel][i] > max_strain or data[channel][i] < min_strain:
            data[channel][i] = 0

    if duration_seconds > 60:
        time_vector = np.linspace(0, duration_seconds / 60, total_samples)
        time_label = 'Time (minutes)'
    else:
        time_vector = np.linspace(0, duration_seconds, total_samples)
        time_label = 'Time (seconds)'
    
    plt.figure(figsize=(15, 5))
    plt.plot(time_vector, data[channel][:total_samples], lw=0.5)
    plt.xlabel(time_label)
    plt.ylabel('Strain')
    plt.title(f'Channel {channel} Strain Over {duration_seconds} Seconds')
    plt.grid(True)
    print("saving plot...")
    plt.savefig(f"single_channel_plot_{channel}_{duration_seconds}_seconds.png")
    print("plot saved...")
    plt.show()

def plot_hdas_from_file(file_path, channel=1500, duration_seconds=600):
    bins = np.sort(glob.glob(file_path))

    for i in tqdm(range(len(bins))):
        hdas_data = HDAS(bins[i], load=True)
        if i == 0:
            combined_data = hdas_data.Data
        else:
            combined_data = np.concatenate((combined_data, hdas_data.Data), axis=1)

    plot_hdas_data(combined_data, channel, duration_seconds)

# Example usage
plot_hdas_from_file("AK_Data/2022_05_19_02*", channel=1500, duration_seconds=60)
