import matplotlib.pyplot as plt
import numpy as np
from lib.data_handler import spiketrain_handler
from lib.data_handler import hd
#from lib.plots import save_figure
import settings
import os

# Function to plot the spike trains
def plot_spiketrains(spike_trains):

    # Plot the raster plot
    plt.figure(figsize=(20, 10))

    spacing = 10  # Increase spacing between rows
    for i, train in enumerate(spike_trains):
        plt.scatter(train, [i * spacing] * len(train), marker='|', color='black')  # Use a vertical marker for spikes

    plt.xlabel("Time in seconds")
    plt.ylabel("Electrode")
    plt.title("Raster Plot")
    # Update y-ticks with new spacing
    plt.yticks([i * spacing for i in range(len(spike_trains))], labels=[f"{i+1}" for i in range(len(spike_trains))])

    plt.tight_layout()  # Ensure labels fit within the figure    
    return plt.gcf()


if __name__ == '__main__':

    # Iterate over all subfolders and files
    for root, dirs, files in os.walk(settings.PATH_DATA_FOLDER):
        for file in files:
            file_path = os.path.join(root, file)
            current_subfolder = os.path.basename(root)
            file_name = file.replace("mat", "jpg")
            path_target = os.path.join(settings.PATH_RESULTS_FOLDER, settings.FOLDER_NAME_RASTERPLOTS, current_subfolder, file_name)
  
            # check if the file already exists
            if os.path.isfile(path_target):
                print(f"File {path_target} already exists. Skipping...")
                continue

            # Load the spike trains
            TS, AMP, t, fs, time, date = spiketrain_handler.read_dr_cell_file(file_path)
            # convert dr_cell TS file to elephant spike train file
            spike_trains_elephant = spiketrain_handler.convert_to_elephant(TS, t_stop=t)

            fig = plot_spiketrains(spike_trains_elephant)

            # Save the figure
            hd.save_figure(fig, path_target)
            plt.close(fig)