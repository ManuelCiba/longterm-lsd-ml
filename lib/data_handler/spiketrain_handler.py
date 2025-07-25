import os
import itertools  # generate all parameter combinations  for parameter study
from pymatreader import read_mat
import glob
import quantities as pq
import elephant.conversion as conv
import neo
import matplotlib.pyplot as plt
import numpy as np
import warnings


def read_dr_cell_file(full_path):
    dr_cell_struct = read_mat(full_path)

    # convert to elephant format
    TS, AMP, t, fs, time, date = _extract_dr_cell_struct(dr_cell_struct)

    return TS, AMP, t, fs, time, date


def _extract_dr_cell_struct(dr_cell_struct):
    try:
        TS = dr_cell_struct['SPIKEZ']['TS']
        AMP = dr_cell_struct['SPIKEZ']['AMP']
        t = dr_cell_struct['SPIKEZ']['PREF']['rec_dur']
        fs = dr_cell_struct['SPIKEZ']['PREF']['SaRa']
        time = dr_cell_struct['SPIKEZ']['PREF']['Time']
        date = dr_cell_struct['SPIKEZ']['PREF']['Date']
    except:
        TS = dr_cell_struct['temp']['SPIKEZ']['TS']
        AMP = dr_cell_struct['temp']['SPIKEZ']['AMP']
        t = dr_cell_struct['temp']['SPIKEZ']['PREF']['rec_dur']
        fs = dr_cell_struct['temp']['SPIKEZ']['PREF']['SaRa']
        time = dr_cell_struct['temp']['SPIKEZ']['PREF']['Time']
        date = dr_cell_struct['temp']['SPIKEZ']['PREF']['Date']
    return TS, AMP, t, fs, time, date


def convert_to_elephant(TS, t_stop=600):
    spiketrains_elephant = []
    
    if len(TS.shape) == 1:
        # If TS is a 1D array, convert it to a 2D array with one column
        TS = TS.reshape(-1, 1).transpose()

    for i in range(TS.shape[1]):
        spiketrain = TS[:, i]
        
        # Set negative values to NaN
        spiketrain[spiketrain < 0] = np.nan
        
        spiketrains_elephant.append(neo.SpikeTrain(spiketrain, t_stop=t_stop, units='s', sort=True))
    return spiketrains_elephant


def bin_spike_trains(spiketrains, bin_size):

    bst = conv.BinnedSpikeTrain(spiketrains, bin_size=bin_size)
    #bst.to_array()
    #bst.to_bool_array()
    bst_binary = bst.binarize()

    return bst_binary

def convert_BinnedSpikeTrainView_to_BinnedSpikeTrain(bst_view):
    """
    Convert a BinnedSpikeTrainView to a full BinnedSpikeTrain.
    This is necessary because some methods require the full BinnedSpikeTrain.
    """
    return conv.BinnedSpikeTrain(
        bst_view.to_array(),
        bin_size=bst_view.bin_size,
        t_start=bst_view.t_start,
        t_stop=bst_view.t_stop,
    )

# Convert BinnedSpikeTrain to a list of neo.SpikeTrains
def OLD_binned_to_spiketrains(binned_spiketrain):
    spike_matrix = binned_spiketrain.to_array()  # Get binned data as a binary matrix
    spiketrains = []
    bin_edges = np.arange(
        binned_spiketrain.t_start.magnitude,
        binned_spiketrain.t_stop.magnitude + binned_spiketrain.bin_size.magnitude,
        binned_spiketrain.bin_size.magnitude,
    ) * binned_spiketrain.t_start.units

    # Iterate over each row (spike train) in the binned data
    for neuron_idx, spike_row in enumerate(spike_matrix):
        spike_times = bin_edges[:-1][spike_row > 0]  # Convert bin indices to times
        spiketrains.append(neo.SpikeTrain(
            spike_times, t_start=binned_spiketrain.t_start, t_stop=binned_spiketrain.t_stop
        ))
    
    return spiketrains

def binned_to_spiketrains(binned_spiketrain):
    """
    Converts a binned spike train into a list of neo.SpikeTrain objects.

    Parameters:
    - binned_spiketrain: A binned spike train object.

    Returns:
    - A list of neo.SpikeTrain objects.
    """
    # Convert binned spike train to binary matrix
    spike_matrix = binned_spiketrain.to_array()

    # Calculate bin edges
    bin_edges = np.arange(
        binned_spiketrain.t_start.magnitude,
        binned_spiketrain.t_stop.magnitude + binned_spiketrain.bin_size.magnitude,
        binned_spiketrain.bin_size.magnitude,
    ) * binned_spiketrain.t_start.units

    # Align lengths of bin_edges[:-1] and spike_matrix columns
    n_bins = min(len(bin_edges) - 1, spike_matrix.shape[1])
    bin_edges = bin_edges[:n_bins + 1]  # Ensure bin_edges matches spike_matrix
    spike_matrix = spike_matrix[:, :n_bins]  # Trim spike_matrix if necessary

    spiketrains = []

    # Convert binary matrix rows to spike trains
    for neuron_idx, spike_row in enumerate(spike_matrix):
        spike_times = bin_edges[:-1][spike_row > 0]  # Convert bin indices to times
        spiketrains.append(neo.SpikeTrain(
            spike_times,
            t_start=binned_spiketrain.t_start,
            t_stop=binned_spiketrain.t_stop
        ))

    return spiketrains

def elephant_spiketrains_to_numpy(spike_trains_elephant):
    # Find the maximum spike train length
    max_length = max(len(st) for st in spike_trains_elephant)
    # Pad each spike train to the maximum length
    ts = np.array([
        np.pad(st.magnitude, (0, max_length - len(st.magnitude)), constant_values=np.nan)
        for st in spike_trains_elephant
        ])
    # transpose so that channels are columns not rows
    ts = ts.T
    return ts

def elephant_spiketrains_to_list(spike_trains_elephant):
    ts = elephant_spiketrains_to_numpy(spike_trains_elephant)
    ts_list = [ts[:, col] for col in range(ts.shape[1])]
    return ts_list



def plot_spike_trains(spiketrains):
    spiketrains = _convert_to_numpy_and_remove_nan(spiketrains)

    # Note: elephant uses different format than DrCell (rows and columns are switched)
    for i in range(spiketrains.shape[0]):
        spiketrain = spiketrains[i, :]
        plt.eventplot(spiketrain[spiketrain != 0], lineoffsets=i, linelengths=0.75, color='black')


def plot_binned_spike_trains(binned_spiketrains):

    binned_spiketrains = binned_spiketrains.to_array()

    # Plot each binary signal separately
    for i in range(binned_spiketrains.shape[0]):
        plt.step(np.arange(len(binned_spiketrains[i])), binned_spiketrains[i] + i, where='post', label=f'Spike Train {i + 1}')


def _convert_to_numpy_and_remove_nan(spiketrains):
    spiketrains = np.asarray(spiketrains)
    spiketrains_new = np.nan_to_num(spiketrains)
    return spiketrains_new
