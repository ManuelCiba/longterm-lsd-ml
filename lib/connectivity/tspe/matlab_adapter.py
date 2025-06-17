from typing import Optional
from pathlib import Path

import matlab.engine
from elephant.conversion import BinnedSpikeTrain
import numpy as np
import numpy.typing as npt

from lib.connectivity.tspe.io import convert_to_sdf


def total_spiking_probability_edges_matlab(
    spike_trains: BinnedSpikeTrain,
    a: Optional[npt.ArrayLike] = None,
    b: Optional[npt.ArrayLike] = None,
    c: Optional[npt.ArrayLike] = None,
    max_delay: int = 25,
    normalize: bool = False,
    matlab_engine: Optional[matlab.engine.MatlabEngine] = None,
    tspe_path: Optional[Path] = None,
):
    # Get tspe matlab function path
    if not tspe_path:
        tspe_path = Path(__file__).parent.parent / "reference"

    if not (tspe_path / "TSPE.m").exists():
        raise FileNotFoundError(tspe_path / "TSPE.m")

    # Start MatlabEngine if no engine was given
    if not matlab_engine:
        matlab_engine = matlab.engine.start_matlab()
        own_matlab_engine = True
    else:
        own_matlab_engine = False

    # Change path to tspe_path
    old_path = matlab_engine.pwd()
    matlab_engine.cd(str(tspe_path.absolute()))

    # Convert options
    if a is not None:
        a = np.array(a,dtype=np.float64)
    else:
        a = np.array([])

    if b is not None:
        b = np.array(b,dtype=np.float64)
    else:
        b = np.array([])

    if c is not None:
        c = np.array(c,dtype=np.float64)
    else:
        c = np.array([])

    if max_delay is not None:
        d = np.float64(max_delay)
    else:
        d = np.array([])

    if normalize:
        flag_normalize = True
    else:
        flag_normalize = np.array([])

    # Convert spike_train_data into SpikeDataFormat
    sdf = convert_to_sdf(spike_trains)

    # Call TSPE-Function via matlab
    connectivity_matrix, delay_matrix = matlab_engine.TSPE(sdf,d,a,b,c,flag_normalize,nargout=2)

    # Handle return types
    connectivity_matrix = np.array(connectivity_matrix)
    delay_matrix = np.array(delay_matrix)

    # Restore old path
    matlab_engine.cd(old_path)

    # Quit matlab if own session was started
    if own_matlab_engine:
        matlab_engine.quit()

    return connectivity_matrix, delay_matrix
