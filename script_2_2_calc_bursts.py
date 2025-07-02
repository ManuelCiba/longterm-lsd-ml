import numpy as np
import os
import settings
import itertools 
import warnings
from lib.data_handler import spiketrain_handler
from lib.data_handler import hd
import quantities as pq
import pandas as pd

SOURCE_PATH = os.path.join(settings.PATH_RESULTS_FOLDER, settings.FOLDER_NAME_split_data)
TARGET_PATH = os.path.join(settings.PATH_RESULTS_FOLDER, settings.FOLDER_NAME_feature_bursts)


def detect_bursts(spike_times, isi_threshold_factor=0.25, min_spikes_in_burst=3):
    if len(spike_times) < min_spikes_in_burst:
        return []
    spike_times = np.sort(spike_times)
    isis = np.diff(spike_times)
    median_isi = np.median(isis)
    isi_threshold = isi_threshold_factor * median_isi

    burst_start_indices = []
    burst_end_indices = []
    in_burst = False
    burst_start = 0

    for i, isi in enumerate(isis):
        if isi <= isi_threshold:
            if not in_burst:
                in_burst = True
                burst_start = i
        else:
            if in_burst:
                in_burst = False
                burst_end = i + 1
                if (burst_end - burst_start) >= min_spikes_in_burst:
                    burst_start_indices.append(burst_start)
                    burst_end_indices.append(burst_end)

    if in_burst:
        burst_end = len(spike_times)
        if (burst_end - burst_start) >= min_spikes_in_burst:
            burst_start_indices.append(burst_start)
            burst_end_indices.append(burst_end)

    bursts_times = [(spike_times[start], spike_times[end - 1]) for start, end in zip(burst_start_indices, burst_end_indices)]
    return bursts_times

def detect_network_bursts(bursts_per_channel, min_active_bursts=3, min_duration=0.1):
    """
    Erkennt Network-Bursts, wenn mindestens min_active_bursts gleichzeitig aktiv sind
    und die Dauer mindestens min_duration Sekunden beträgt.
    """
    events = []
    for ch, bursts in bursts_per_channel.items():
        for start, end in bursts:
            events.append((start, +1))
            events.append((end, -1))
    events.sort(key=lambda x: (x[0], -x[1]))

    active = 0
    network_start = None
    network_bursts = []
    for time_pt, delta in events:
        active += delta
        if active >= min_active_bursts and network_start is None:
            network_start = time_pt
        elif active < min_active_bursts and network_start is not None:
            network_end = time_pt
            if (network_end - network_start) >= min_duration:
                network_bursts.append((network_start, network_end))
            network_start = None
    # letzten Network-Burst abschließen
    if network_start is not None:
        end_time = events[-1][0]
        if (end_time - network_start) >= min_duration:
            network_bursts.append((network_start, end_time))
    return network_bursts

def calculate_bursts(source_path, target_path):
    # Load the spike trains
    binned_spike_trains_elephant = hd.load_pkl_as_list(source_path)
    spike_trains_elephant = spiketrain_handler.binned_to_spiketrains(binned_spike_trains_elephant)

    # Check if spike_trains_elephant is empty
    all_empty = all((not element.size) if isinstance(element, np.ndarray) else not bool(element) for element in spike_trains_elephant)
    if all_empty:
        print(f"No spikes found in {source_path}. Skipping calculation.")
        return

    # convert elephant spike trains to numpy array
    #ts_list = spiketrain_handler.elephant_spiketrains_to_list(spike_trains_elephant)

    rec_dur = binned_spike_trains_elephant.t_stop - binned_spike_trains_elephant.t_start

    # calculate bursts:
    fr_list = []
    burst_list = []
    burst_rate_list = []
    burst_duration_list = []
    for ts in spike_trains_elephant:
        fr_list.append(len(ts) / rec_dur)
        bursts = detect_bursts(np.array(ts))
        burst_rate = len(bursts) / rec_dur
        burst_duration = [b[1] - b[0] for b in bursts]
        burst_list.append(bursts)
        burst_rate_list.append(burst_rate)
        burst_duration_list.append(np.mean(burst_duration))
    
    # Calculate mean firing rate
    fr_np = np.array(fr_list)
    # Filter out zeros (omit zero-magnitude values)
    non_zero_values = fr_np[fr_np != 0 * pq.s]
    # Calculate mean burst rate
    mfr = np.mean(non_zero_values)

    # Calculate mean burst rate
    burst_rate_np = np.array(burst_rate_list)
    # Filter out zeros (omit zero-magnitude values)
    non_zero_values = burst_rate_np[burst_rate_np != 0 * pq.s]
    # Calculate mean burst rate
    mbr = np.mean(non_zero_values)

    # Calculate mean burst duration
    mbd = np.nanmean(burst_duration_list)

    # save all bursts features into one dataframe
    all_burst_features = {
        "Mean Firing Rate /Hz": [mfr],
        "Mean Burst Rate /Hz": [mbr],
        "Mean Burst Duration /s": [mbd]
                }
    df = pd.DataFrame(all_burst_features)

    # Save the results 
    hd.save_df_as_csv(df, target_path)


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
                files.sort()

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
                        print(f"Calculating bursts: {target_path}")
                        calculate_bursts(source_path, target_path)