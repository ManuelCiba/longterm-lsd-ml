import numpy as np
import numpy.typing as npt

import matplotlib.pyplot as plt

def mean_squared_error(a,b):
        return np.nanmean(np.power(a-b,2))

def apply_threshold(connectivity_matrix: npt.NDArray,threshold:int):
    mask = np.abs(connectivity_matrix) <= threshold
    connectivity_matrix_threshold = connectivity_matrix.copy()
    connectivity_matrix_threshold[mask] = 0
    return connectivity_matrix_threshold

def classify_connections(connectivity_matrix:npt.NDArray,threshold:int):
    connectivity_matrix_binarized = connectivity_matrix.copy()

    mask_excitatory = connectivity_matrix_binarized > threshold
    mask_inhibitory = connectivity_matrix_binarized < -threshold

    mask_left = ~ (mask_excitatory + mask_inhibitory)

    connectivity_matrix_binarized[mask_excitatory] = 1
    connectivity_matrix_binarized[mask_inhibitory] = -1
    connectivity_matrix_binarized[mask_left] = 0

    return connectivity_matrix_binarized


def confusion_matrix(estimate, original, threshold: int = 1):
    """
    Definition:
        - TP: Matches of connections are True Positive
        - FP: Mismatches are False Positive,
        - TN: Matches for non-existing synapses are True Negative
        - FN: mismatches are False Negative.
    """
    if not np.all(np.isin([-1,0,1], np.unique(estimate))):
        estimate = classify_connections(estimate,threshold)
    if not np.all(np.isin([-1,0,1], np.unique(original))):
        original = classify_connections(original,threshold)

    TP = (np.not_equal(estimate,0) & np.not_equal(original,0)).sum()

    TN = (np.equal(estimate,0) & np.equal(original, 0)).sum()

    FP = (np.not_equal(estimate,0) & np.equal(original, 0)).sum()

    FN = (np.equal(estimate, 0) & np.not_equal(original,0)).sum()

    return TP, TN, FP, FN

def sensitivity(TP:int, TN:int, FP:int, FN:int):
    TPR = TP / (TP + FN)
    return TPR

def specificity(TP:int, TN:int, FP:int, FN:int):
    TNR = TN / (TN + FP)
    return TNR

def precision(TP:int, TN:int, FP:int, FN:int):
    PPV = TP / (TP + FP)
    return PPV

def fall_out(TP:int, TN:int, FP:int, FN:int):
    FPR = FP / (FP + TN)
    return FPR


def roc_curve(estimate,original):
    tpr_list = []
    fpr_list = []

    max_threshold = max(np.max(np.abs(estimate)),1)

    thresholds = np.linspace(max_threshold,0,30)

    for t in thresholds:
        conf_matrix = confusion_matrix(estimate,original,threshold=t)

        tpr_list.append(sensitivity(*conf_matrix))
        fpr_list.append(fall_out(*conf_matrix))

    auc = np.trapz(tpr_list, fpr_list)

    return tpr_list, fpr_list, thresholds, auc


def plot_roc_curve(estimate,original):
    plt.figure()
    tpr_rate, fpr_rate, _, auc = roc_curve(estimate,original)
    plt.plot(fpr_rate,
             tpr_rate,
             'b',linestyle='--',marker='o',lw=2,
             label='ROC curve',
             clip_on=False)
    plt.plot([0,1],[0,1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title(f'ROC curve, AUC = {auc:.2} ')
    plt.legend(loc='lower right')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
