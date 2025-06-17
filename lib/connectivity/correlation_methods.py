# different methods to calculate the connectivity/correlation matices
# between two or more parallel spike trains.
#
# Input data format:
# bst = binned spike train matrix format used by the Elephant python library
# (matrix dimension: #channels x #spikes)
# x, y: a binned spike train
#
# Output data format:
# 2D connectivity matrix, dimension: #channels x # channels
#
# correlations are calculated as "pairwise" = all possible combinations of two pairs
# or as "global" = the whole bst matrix is used at once to generate the correlation matrix

from scipy import stats
from pyinform.transferentropy import transfer_entropy
from pyinform.mutualinfo import mutual_info
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.covariance import EmpiricalCovariance
from sklearn.covariance import GraphicalLassoCV

from lib.connectivity.tspe import tspe


def tspe_pairwise(bst):
    connectivity_matrix, delay_matrix = tspe.tspe(bst)
    return connectivity_matrix


def spearman(x, y):
    value, _ = stats.spearmanr(x, y)
    return value


def spearman_pairwise(bst):
    M = _calculate_pairwise_correlation(bst, spearman)
    return M


def te(x, y):
    value = transfer_entropy(x, y, k=1)
    return value


def te_pairwise(bst):
    M = _calculate_pairwise_correlation(bst, te)
    return M


def mi(x, y):
    value = mutual_info(x, y)
    return value


def mi_pairwise(bst):
    M = _calculate_pairwise_correlation(bst, mi)
    return M


def pearson(x, y):
    value = np.corrcoef(x, y)[0, 1]
    return value


def pearson_pairwise(bst):
    M = _calculate_pairwise_correlation(bst, pearson)
    return M


def ncc(x, y):
    norm_x = np.linalg.norm(x)
    x = x / norm_x
    norm_y = np.linalg.norm(y)
    y = y / norm_y
    correlogram = np.correlate(x, y, mode='full')
    # use maximum correlation as final value.
    # note: delay is not used yet
    value = np.max(correlogram)
    return value


def ncc_pairwise(bst):
    M = _calculate_pairwise_correlation(bst, ncc)
    return M


def ledoit_wolf(x, y):
    # put the x and y array into a matrix
    M = np.column_stack((x, y))

    estimator = LedoitWolf()
    estimator.fit(M)
    value = np.array(estimator.covariance_)[0, 1]
    return value


def ledoit_wolf_pairwise(bst):
    M = _calculate_pairwise_correlation(bst, ledoit_wolf)
    return M


def ledoit_wolf_global(bst):
    estimator = LedoitWolf()
    estimator.fit(bst.to_array().T)
    M = np.array(estimator.covariance_)
    return M


def canonical(x, y):
    # put the x and y array into a matrix
    M = np.column_stack((x, y))

    estimator = EmpiricalCovariance()
    estimator.fit(M)
    value = np.array(estimator.covariance_)[0, 1]
    return value


def canonical_pairwise(bst):
    M = _calculate_pairwise_correlation(bst, canonical)
    return M


def canonical_global(bst):
    estimator = EmpiricalCovariance()
    estimator.fit(bst.to_array().T)
    M = np.array(estimator.covariance_)
    return M


def graph_lasso(x, y):
    # put the x and y array into a matrix
    M = np.column_stack((x, y))

    estimator = GraphicalLassoCV()
    try:
        estimator.fit(M)
        value = np.array(estimator.covariance_)[0, 1]
    except Exception:
        print("Warning: GraphicalLasso x-y-pair is zero!")
        value = np.nan
    return value


def graph_lasso_pairwise(bst):
    M = _calculate_pairwise_correlation(bst, graph_lasso)
    return M


def graph_lasso_global(bst):
    estimator = GraphicalLassoCV()
    estimator.fit(bst.to_array().T)
    # .precision_ is the inverse of covariance and is better for graphs
    # .covariance_ is good if interested in the pairwise relationship
    M = np.array(estimator.precision_)

    return M


def covariance_global(bst):
    M = np.cov(bst.to_array().T, rowvar=False)
    return M


##############################
# General function for pairwise calculation, can be used
# with any correlation function "func", e.g. func=ncc()
#
# Input: bst = binned spike train in the elephant format
# Note! dimensions are like that: num_channels x num_spikes
##############################

def _calculate_pairwise_correlation(bst, func):
    bst_np = bst.to_array()
    channels = bst.shape[0]
    M = np.zeros((channels, channels))
    M[M == 0] = np.nan  # set all zeros to NaN
    for i in range(channels):
        x = bst_np[i, :]
        for j in range(channels):
            y = bst_np[j, :]

            M[i][j] = func(x, y)

    return M