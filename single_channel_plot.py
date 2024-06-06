import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
from functions.hdas_class import HDAS

def plot_hdas_data(data, sample_rate=100, channel=1500):
 
    num_samples = len(data[1500])

    x_axis_values1 = []
    x_axis_values2 = []

    for i in range(0, int(num_samples/30000) + 1):
        x_axis_values1.append(i * 30000)
        x_axis_values2.append(i * 5)                                                                                                                                                                                                               

    plt.figure(figsize=(15, 5))
    plt.plot(data[1500], lw=0.1)
    plt.xlabel('Time (minutes)')
    plt.ylabel('Strain')
    plt.title(f'Channel {channel} Strain Over Time')
    plt.xticks(x_axis_values1,x_axis_values2)   
    plt.grid(True)
    plt.savefig("single_channel_plot.png")
    plt.show()

def plot_hdas_from_file(file_path, num_channels=4992, sample_rate=100, channel=1500):

    bins = np.sort(glob.glob(file_path))

    for i in tqdm(range(len(bins))):

        hdas_data = HDAS(bins[i], load=True)

        if i == 0:
            combined_data = hdas_data.Data
        else:
            combined_data = np.concatenate((combined_data, hdas_data.Data), axis=1)  # concatenate along samples

    # Plot the specified channel
    plot_hdas_data(combined_data, sample_rate, channel)

# Example usage
plot_hdas_from_file("AK_Data/2022_05_19_02*", channel=1500)
