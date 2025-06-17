from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import numpy as np

from elephant.conversion import BinnedSpikeTrain
from neo import SpikeTrain
from quantities import Quantity, millisecond
from scipy.io import loadmat

DEFAULT_BIN_SIZE = 1 * millisecond

def load_spike_train_mat(
    path: Union[Path, str],
    bin_size: Optional[Quantity] = None,
    t_stop: Optional[Quantity] = None,
    format: Literal['SPIKEZ','SIMULATED'] = 'SPIKEZ'
) -> Tuple[BinnedSpikeTrain,Optional[np.ndarray]]:
    if format=='SPIKEZ':
        spiketrains = load_spike_train_spikez(path,bin_size,t_stop)
        original_data = None
    elif format =='SIMULATED':
        spiketrains, original_data = load_spike_train_simulated(path,bin_size,t_stop)
    else:
        raise NotImplementedError(f'Format {format} is not implemented yet. Valid formats include SPIKEZ, SIMULATED')

    return spiketrains, original_data

def load_spike_train_simulated(
    path: Union[Path, str],
    bin_size: Optional[Quantity] = None,
    t_stop: Optional[Quantity] = None,
) -> Tuple[BinnedSpikeTrain, np.ndarray]:
    if isinstance(path, str):
        path = Path(path)

    if not bin_size:
        bin_size = DEFAULT_BIN_SIZE

    data = loadmat(path, simplify_cells=True)["data"]

    if "asdf" not in data:
        raise ValueError('Incorrect Dataformat: Missing spiketrain_data in "asdf"')

    spiketrain_data = data["asdf"]

    # Get number of electrodesa and recording_duration from last element of data array
    n_electrodes, recording_duration_ms = spiketrain_data[-1]
    recording_duration_ms = recording_duration_ms * millisecond

    # Create spiketrains
    spiketrains = []
    for spiketrain_raw in spiketrain_data[0:n_electrodes]:
        spiketrains.append(
            SpikeTrain(
                spiketrain_raw * millisecond,
                t_stop= recording_duration_ms,
            )
        )

    spiketrains = BinnedSpikeTrain(spiketrains, bin_size=bin_size, t_stop = t_stop or recording_duration_ms)

    # Load original_data
    original_data = data['SWM'].T

    return spiketrains, original_data


def load_spike_train_spikez(
    path: Union[Path, str],
    bin_size: Optional[Quantity] = None,
    t_stop: Optional[Quantity] = None,
) -> BinnedSpikeTrain:

    if isinstance(path, str):
        path = Path(path)

    if not bin_size:
        bin_size = DEFAULT_BIN_SIZE

    data = loadmat(path, simplify_cells=True)["SPIKEZ"]

    if "TS" not in data:
        raise ValueError('Incorrect dataformat: No timestamps in data under "TS"')

    spiketrain_timestamps = np.transpose(data["TS"])

    if "PREF" not in data:
        raise ValueError('Incorrect dataformat: No metadata in data under "PREF"')

    spiketrain_metadata = data["PREF"]

    # Get number of electrodesa and recording_duration from last element of data array
    n_electrodes = len(spiketrain_timestamps)
    recording_duration_ms = spiketrain_metadata["rec_dur"] * millisecond

    # Create spiketrains
    spiketrains = []
    for spiketrain_raw in spiketrain_timestamps[0:n_electrodes]:
        if not spiketrain_raw.any():
            continue
        spiketrains.append(
            SpikeTrain(spiketrain_raw * millisecond, t_stop=recording_duration_ms)
        )

    spiketrains = BinnedSpikeTrain(spiketrains, bin_size=bin_size, t_stop=t_stop or recording_duration_ms)

    return spiketrains

def convert_to_sdf(spike_train_data: BinnedSpikeTrain):
    n_electrons, recording_duration_ms = spike_train_data.shape
    sdf_int32 = spike_train_data.spike_indices
    # Convert to float64
    sdf = []
    for sdf_electrode in sdf_int32:
        sdf.append(sdf_electrode.astype(np.float64))

    sdf.append(np.array([n_electrons,recording_duration_ms],dtype=np.float64))

    return sdf
