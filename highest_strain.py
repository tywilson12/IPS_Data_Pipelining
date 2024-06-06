import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
from functions.hdas_class import HDAS

def plot_hdas_data(data, sample_rate=100):
    num_samples = len(data[1500])

    highest_strain_data = []

    for i in tqdm(range(len(data[0]))):
        max_strain = -10e10
        strongest_channel = 0
        for j in range(len(data)):
            if data[j][i] > max_strain:
                max_strain = data[j][i]
                strongest_channel = j
        highest_strain_data.append(max_strain)

    avg_strain = np.mean(highest_strain_data)
    print(f"Average Strain: {avg_strain}")

    x_axis_values1 = []
    x_axis_values2 = []

    for i in range(0, int(num_samples/30000) + 1):
        x_axis_values1.append(i * 30000)
        x_axis_values2.append(i * 5)                                                                                                                                                                                                               

    plt.figure(figsize=(15, 5))
    plt.plot(highest_strain_data, lw=0.1)
    plt.xlabel('Time (minutes)')
    plt.ylabel('Strain')
    plt.title(f'Highest Strain on Run Over Time')
    plt.xticks(x_axis_values1, x_axis_values2)   
    plt.grid(True)
    print("Saving the plot...")
    plt.savefig("testplot_highest_strain.png")
    print("Plot saved.")
    plt.show()

def plot_hdas_from_file(file_path, num_channels=4992, sample_rate=100):
    bins = np.sort(glob.glob(file_path))
    combined_data = None

    for i in tqdm(range(len(bins))):
        hdas_data = HDAS(bins[i], load=True)

        if i == 0:
            combined_data = hdas_data.Data
        else:
            combined_data = np.concatenate((combined_data, hdas_data.Data), axis=1)

    if combined_data is not None:
        print("Plotting data...")
        plot_hdas_data(combined_data, sample_rate)
        print("Plotting complete.")
    else:
        print("No data to plot.")

plot_hdas_from_file("AK_Data/2022_05_19_02*")
