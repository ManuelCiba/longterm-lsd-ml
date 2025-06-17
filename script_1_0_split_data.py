# This script splits spike trains into windows and saves them as CSV files.
# Before splitting, it applies a minimum firing rate filter to remove spike trains with low firing rates.

import os
import itertools  # generate all parameter combinations  for parameter study
import glob
import quantities as pq
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import neo

from lib.data_handler import spiketrain_handler
from lib.data_handler import folder_structure
from lib.data_handler import hd
import settings


TARGET_DATA_FOLDER = settings.FOLDER_NAME_split_data


def apply_minimum_firing_rate_filter(spiketrains, min_fr=0.1 * pq.Hz):
    """
    Filter out spike trains with a firing rate below the specified minimum firing rate.
    """
    filtered_spiketrains = []
    for st in spiketrains:
        # Calculate firing rate as spikes per second
        duration = (st.t_stop - st.t_start).rescale(pq.s)  # Convert to seconds
        firing_rate = len(st) / duration
        if firing_rate >= min_fr:
            filtered_spiketrains.append(st)
        else:
            # Create a "NaN" spike train with the same time range
            nan_spiketrain = neo.SpikeTrain(
                np.array([np.nan]) * pq.s,  # Replace spikes with NaN
                t_start=st.t_start,
                t_stop=st.t_stop,
                units=st.units
            )
            filtered_spiketrains.append(nan_spiketrain)
            print(f"Spike train with firing rate {firing_rate} Hz filtered out (below {min_fr} Hz).")
    return filtered_spiketrains

def split_spiketrains_and_save_results(spiketrains, bin_size, window_size, window_overlap, result_folder_path, time, date):

    # bin spike trains
    bst = spiketrain_handler.bin_spike_trains(spiketrains, bin_size)

    # split binned spike trains into windows
    recording_size = bst.t_stop
    step = window_size * (1 - window_overlap)
    window_start_list = list(range(0, int(recording_size), int(step)))
    window_end_list = list(range(int(window_size), int(recording_size) + int(step), int(step)))

    num_windows = min([len(window_start_list), len(window_end_list)])
    for w in range(num_windows):

        # define file name
        full_path = define_full_file_name(result_folder_path, date, time, w)

        # only calculate if result does not exist yet
        if not os.path.isfile(full_path):
            print("Splitting spike trains... " + full_path)

            # get binned spike train for the current window
            bst_w = bst.time_slice(t_start=window_start_list[w] * pq.s, t_stop=window_end_list[w] * pq.s)

            # save split spike trains as an HDF5 file
            bst_w_full = spiketrain_handler.convert_BnnedSpikeTrainView_to_BinnedSpikeTrain(bst_w)
            hd.save_list_as_pkl(bst_w_full, full_path)

            # plot spike trains and save
            if settings.FLAG_PLOT:
                plot_spiketrains_and_save(bst_w, full_path)
        else:
            print("Already processed: " + full_path)


def plot_spiketrains_and_save(bst, full_path):

    fig, ax = plt.subplots()  # Create a figure and axis object

    # Heatmap for binary spike train matrix
    binary_data = bst.to_bool_array()
    sns.heatmap(binary_data, cmap='binary', cbar=False, xticklabels=False)
    ax.set_xticks([0, binary_data.shape[1] - 1])  # Start at 0, end at the number of columns (time bins)
    ax.set_xticklabels([str(bst.t_start), str(bst.t_stop)])  # Use t_start and t_stop as labels
    plt.xlabel('Time bins')
    plt.ylabel('Electrode index')
    plt.title('Binned Spike Trains')
    hd.save_figure(fig, full_path.replace("csv", "jpg"))
    fig.clear()


def define_full_file_name(result_folder, date, time, w):
    file_name = 'window' + '{:02}'.format(w)
    date_time_folder = date + "_" + time
    full_path = os.path.join(result_folder, date_time_folder, file_name)
    return full_path




if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    # define parameter
    bin_sizes = settings.BIN_SIZES
    window_sizes = settings.WINDOW_SIZES
    window_overlaps = settings.WINDOW_OVERLAPS
    chip_names = settings.WELLS_CTRL + settings.WELLS_LSD  # Combine control and LSD wells

    # Generate all combinations of parameters
    parameter_combinations = list(itertools.product(bin_sizes, window_sizes, window_overlaps, chip_names))

    # Base folder for storing results
    base_folder = os.path.join(settings.PATH_RESULTS_FOLDER, TARGET_DATA_FOLDER)

    for combination in parameter_combinations:
        bin_size, window_size, window_overlap, chip_name = combination

        # Create a folder structure based on the parameter combination
        result_folder = os.path.join(base_folder, f"bin_{bin_size}",
                                     f"window_{window_size}", f"overlap_{window_overlap}", chip_name)


        # 1) load file
        # define full path of selected chip
        path_chip = os.path.join(os.getcwd(), settings.PATH_DATA_ISMAEL, chip_name)
        
        # get file name of all .mat file within folder
        files = glob.glob(path_chip + '/*.mat')
        # Important! Sort file names, as python make wrong order
        files = sorted(files)

        # For all files in the folder
        for file in files:
            # Load the spike trains
            TS, AMP, t, fs, time, date = spiketrain_handler.read_dr_cell_file(file)
            # convert dr_cell TS file to elephant spike train file
            spiketrains_elephant = spiketrain_handler.convert_to_elephant(TS, t_stop=t)

            # 1.1) apply minimum firing rate filter
            spiketrains_filtered = apply_minimum_firing_rate_filter(spiketrains_elephant, min_fr=settings.MIN_FR)

            # 2) split spike trains
            split_spiketrains_and_save_results(spiketrains_filtered, bin_size, window_size, window_overlap, result_folder, time, date)

