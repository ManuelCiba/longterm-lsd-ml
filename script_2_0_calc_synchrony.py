import matplotlib.pyplot as plt
import itertools  # generate all parameter combinations  for parameter study
import numpy as np
from lib.data_handler import spiketrain_handler
from lib.data_handler import hd
import settings
import os
import quantities as pq
import warnings
from elephant.spike_train_synchrony import spike_contrast
from elephant.functional_connectivity import total_spiking_probability_edges as tspe
import viziphant  # visualization for elephant results
import pandas as pd
 
SOURCE_PATH = os.path.join(settings.PATH_RESULTS_FOLDER, settings.FOLDER_NAME_split_data)
TARGET_PATH = os.path.join(settings.PATH_RESULTS_FOLDER, settings.FOLDER_NAME_feature_synchrony)

def calculate_and_plot_and_save_synchrony(source_path, target_path):
    # Load the spike trains
    binned_spike_trains_elephant = hd.load_pkl_as_list(source_path)
    spike_trains_elephant = spiketrain_handler.binned_to_spiketrains(binned_spike_trains_elephant)

    # Calculate synchrony
    df_result, fig = _calculate_synchrony(spike_trains_elephant)

    # Save the figure
    figure_path = target_path.replace("pkl", "jpg")
    hd.save_figure(fig, figure_path)
    plt.close(fig)

    # Save the results
    result_path = target_path.replace("pkl", "csv")
    df_result.to_csv(result_path)

def _calculate_synchrony(st_list):

    # init results, init figure
    df_result = pd.DataFrame()
    plt.figure(figsize=(8, 6))
    fig = plt.gcf()
 
    s, trace = spike_contrast(st_list, return_trace=True)  # return_trace=True: returns not only synchrony value but also curves
    result = trace.synchrony  # save whole synchrony curve, not only max value
    names = ["Spike-contrast (" + str(round(bin_size, 3)) + ")" for bin_size in trace.bin_size]
    df_result = pd.DataFrame(data=[result], columns=names)
    if not np.isnan(s):
        fig = _plot_spike_contrast(trace, st_list)


    return df_result, fig

def _plot_spike_contrast(trace, st_list):
    # Create a heatmap using seaborn
    plt.figure(figsize=(8, 6))
    viziphant.spike_train_synchrony.plot_spike_contrast(trace, spiketrains=st_list, c='gray', s=1)
    # Get the figure handle
    fig = plt.gcf()
    return fig


if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    # define parameter
    bin_sizes = settings.BIN_SIZES
    window_sizes = settings.WINDOW_SIZES
    window_overlaps = settings.WINDOW_OVERLAPS

    # get all chip names
    chip_names = settings.WELLS_CTRL + settings.WELLS_LSD  # Combine control and LSD wells

    # Generate all combinations of parameters
    parameter_combinations = list(itertools.product(bin_sizes, window_sizes, window_overlaps, chip_names))

    # loop through each combination of parameters
    for combination in parameter_combinations:
        bin_size, window_size, window_overlap, chip_name = combination

        # Create a folder structure based on the parameter combination
        current_path = os.path.join(SOURCE_PATH, f"bin_{bin_size}",
                                     f"window_{window_size}", f"overlap_{window_overlap}", chip_name)
  
        # List all directories in the current path
        folders = [entry for entry in os.listdir(current_path) if os.path.isdir(os.path.join(current_path, entry))]

        # Loop through each folder and process the files
        for folder in folders:
            # Define the full path to the folder
            folder_path = os.path.join(current_path, folder)
            result_path = folder_path.replace(SOURCE_PATH, TARGET_PATH)

            # List all .pkl files in the folder
            files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]

            # Process each .pkl file (= spike train file)
            for file in files:
                source_file = os.path.join(folder_path, file)
                target_file = os.path.join(result_path, file)

                # Test if target file already exists
                if os.path.exists(target_file):
                    print(f"Already processed: {target_file}")
                    continue
                else:
                    # Calculate synchrony and save results
                    print(f"Calculating synchrony: {source_file}")
                    calculate_and_plot_and_save_synchrony(source_file, target_file)