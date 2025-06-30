import os
import pandas as pd
import h5py
import numpy as np
import pickle

def save_spiketrains_as_csv(bst, full_path):
    # create directory
    result_folder = os.path.dirname(full_path)
    os.makedirs(result_folder, exist_ok=True)

    # Extract the binned spike train data (binary matrix)
    binary_data = bst.to_bool_array()

    # Create a Pandas DataFrame to save the binary spike train data
    bin_edges = np.arange(bst.t_start.magnitude, bst.t_stop.magnitude, bst.bin_size.magnitude) * bst.units

    # Create a dictionary to store the spike times for each neuron
    spike_times_dict = {}

    # Iterate through each neuron
    for i, neuron_data in enumerate(binary_data):
        # Find the indices where spikes occur (True values)
        spike_indices = np.where(neuron_data == True)[0]

        # Get the corresponding times for these spike indices
        spike_times = bin_edges[spike_indices]

        # Store the spike times in the dictionary
        spike_times_dict[f'El {i + 1}'] = spike_times.magnitude  # Convert to float for saving in CSV

    # Convert the dictionary to a DataFrame for easy saving to CSV
    df_spike_times = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in spike_times_dict.items()]))

    # Save to CSV
    df_spike_times.to_csv(full_path, index=False)


def load_csv_as_df(full_path, index_col=False):
    # set index_col=0 if the csv file contains row-names
    # header=0 and index_col=0 will use the row and col names like in the csv file
    df_matrix = pd.read_csv(full_path, header=0, index_col=index_col)
    return df_matrix


def save_df_as_csv(df, full_path, index=False):
    # check if the file ends with .csv, if not, add it
    if not full_path.endswith(".csv"):
        # if the file does not end with .csv, add it
        full_path = full_path + ".csv"
        
    # create directory
    result_folder = os.path.dirname(full_path)
    os.makedirs(result_folder, exist_ok=True)

    #df = pd.DataFrame(result)
    #df.to_csv(full_path, sep=',', header=header, index=False)
    df.to_csv(full_path, sep=',', index=index)

def save_list_as_pkl(list, full_path):
    # check if the file ends with .pkl, if not, add it
    if not full_path.endswith(".pkl"):
        # if the file does not end with .pkl, add it
        full_path = full_path + ".pkl"
    
    # create directory
    result_folder = os.path.dirname(full_path)
    os.makedirs(result_folder, exist_ok=True)
    
    # save lists as pickles format
    with open(full_path, 'wb') as f:
        pickle.dump(list, f)

def load_pkl_as_list(full_path):
    with open(full_path, 'rb') as f:
        list = pickle.load(f)
    return list


def save_figure(fig, full_path):

    # create directory
    result_folder = os.path.dirname(full_path)
    os.makedirs(result_folder, exist_ok=True)

    fig.savefig(full_path, bbox_inches='tight', dpi=300)
    print("    Saved: " + full_path)