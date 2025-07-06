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

def calculate_and_plot_and_save_synchrony(source_path, target_path, window_size):
    # Load the spike trains
    binned_spike_trains_elephant = hd.load_pkl_as_list(source_path)
    spike_trains_elephant = spiketrain_handler.binned_to_spiketrains(binned_spike_trains_elephant)

    # Check if spike_trains_elephant is empty
    all_empty = all((not element.size) if isinstance(element, np.ndarray) else not bool(element) for element in spike_trains_elephant)
    if all_empty:
        print(f"No spikes found in {source_path}. Skipping synchrony calculation.")
        return


    # Calculate synchrony
    try:
        df_curve, fig = _calculate_synchrony(spike_trains_elephant, window_size)
    except Exception as e:
        print(f"Error calculating synchrony for {source_path}: {e}")
        return

    # Save the figure
    if settings.FLAG_PLOT:
        figure_path = target_path.replace("csv", "pdf")
        hd.save_figure(fig, figure_path)
    plt.close(fig)

    # Save the results 
    hd.save_df_as_csv(df_curve, target_path)


def _calculate_synchrony(st_list, window_size):

    # init results, init figure
    df_result = pd.DataFrame()
    plt.figure(figsize=(8, 6))
    fig = plt.gcf()
 
    # Define t_start and t_stop for synchrony calculation
    # important to define so that synchrony is calculated for the same bins for all spike trains
    # the same bins are needed to compare synchrony curves since they are used as features for ML
    t_start = st_list[0].t_start.rescale(pq.s)  # start time of the first spike train
    t_stop = t_start + window_size 
    s, trace = spike_contrast(st_list, return_trace=True, t_start=t_start, t_stop=t_stop)  # return_trace=True: returns not only synchrony value but also curves
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
    chip_names = settings.WELLS_SHAM + settings.WELLS_DRUG  # Combine control and DRUG wells

    # Generate all combinations of parameters
    parameter_combinations = list(itertools.product(bin_sizes, window_sizes, window_overlaps))

    # loop through each combination of parameters
    for combination in parameter_combinations:
        bin_size, window_size, window_overlap = combination

        # Create a folder structure based on the parameter combination
        current_path = os.path.join(SOURCE_PATH, f"bin_{bin_size}",
                                     f"window_{window_size}", f"overlap_{window_overlap}")
  
        # get all directories in the current path (=chip names)
        chip_folders = [entry for entry in os.listdir(current_path) if os.path.isdir(os.path.join(current_path, entry))]    
        # sort the chip folders
        chip_folders.sort()

        # Loop through each chip folder
        for chip_folder in chip_folders:

            # Define the full path to the chip folder
            chip_folder_path = os.path.join(current_path, chip_folder)

            # List all directories in the current path
            rec_folders = [entry for entry in os.listdir(chip_folder_path) if os.path.isdir(os.path.join(chip_folder_path, entry))]

            # Loop through each folder (=recordings) and process the files
            for rec_folder in rec_folders:
                # Define the full path to the folder
                rec_folder_path = os.path.join(chip_folder_path, rec_folder)
                result_path = rec_folder_path.replace(SOURCE_PATH, TARGET_PATH)

                # List all .pkl files in the folder
                files = [f for f in os.listdir(rec_folder_path) if f.endswith('.pkl')]

                # Process each .pkl file (= spike train file) and
                # calculate snychrony
                for file in files:
                    source_path = os.path.join(rec_folder_path, file)
                    target_path = os.path.join(result_path, file)
                    target_path = target_path.replace("pkl", "csv")

                    # Test if target file already exists
                    if os.path.exists(target_path):
                        print(f"Already processed: {target_path}")
                        continue
                    else:
                        # Calculate synchrony and save results
                        print(f"Calculating synchrony: {target_path}")
                        calculate_and_plot_and_save_synchrony(source_path, target_path, window_size)