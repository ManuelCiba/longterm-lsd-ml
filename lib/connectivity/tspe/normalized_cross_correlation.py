from typing import List, Iterable, Union

from elephant.conversion import BinnedSpikeTrain
import numpy as np


def normalized_cross_correlation(
    spike_trains: BinnedSpikeTrain,
    delay_times: Union[int, List[int], Iterable[int]] = 0,
) -> np.ndarray:
    r"""normalized cross correlation using std deviation

    Computes the normalized_cross_correlation between all
    Spiketrains inside a BinnedSpikeTrain-Object at a given delay_time

    The underlying formula is:

    .. math::
        NCC_{X\arrY(d)} = \frac{1}{N_{bins}}\sum_{i=-\inf}^{\inf}{\frac{(y_{(i)} - \bar{y}) \cdot (x_{(i-d) - \bar{x})}{\sigma_x \cdot \sigma_y}}}

    """

    n_neurons, n_bins = spike_trains.shape

    # Get sparse array of BinnedSpikeTrain
    spike_trains_array = spike_trains.sparse_matrix

    # Get std deviation of spike trains
    spike_trains_zeroed = spike_trains_array - spike_trains_array.mean(axis=1)
    spike_trains_std = np.std(spike_trains_zeroed,ddof=1,axis=1)
    std_factors = spike_trains_std @ spike_trains_std.T

    # Loop over delay times
    if isinstance(delay_times, int):
        delay_times = [delay_times]
    elif isinstance(delay_times, list):
        pass
    elif isinstance(delay_times, Iterable):
        delay_times = list(delay_times)

    NCC_d = np.zeros((len(delay_times), n_neurons, n_neurons))

    for index, delay_time in enumerate(delay_times):
        # Uses theoretical zero-padding for shifted values,
        # but since $0 \cdot x = 0$ values can simply be omitted
        if  delay_time == 0:
            CC = (
                spike_trains_array[:,:]
                @ spike_trains_array[:,:].transpose()
            )

        elif delay_time > 0:
            CC = (
                spike_trains_array[:, delay_time:]
                @ spike_trains_array[:, : -delay_time].transpose()
            )

        else:
            CC = (
                spike_trains_array[:, :delay_time]
                @ spike_trains_array[:, -delay_time:].transpose()
            )

        # Normalize using std deviation
        NCC = CC / std_factors / n_bins

        # Compute cross correlation at given delay time
        NCC_d[index, :, :] = NCC.todense()  # XXX: mc: convert sparse (COO) matrix to dense, otherwise error

    # Move delay_time axis to back of array
    # Makes index using neurons more intuitive → (n_neuron, n_neuron, delay_times)
    NCC_d = np.moveaxis(NCC_d, 0, -1)

    return NCC_d
