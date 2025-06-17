import os
import quantities as pq

PATH_DATA_ISMAEL = "/home/mc/Documents/Data/LSD/IsmaelLozanoFlores/LongTerm/TS/"
PATH_RESULTS_FOLDER = os.path.join(os.getcwd(), "results")

# Ismael's data:
# LSD (10µM) and CTRL (1% DMSO) well definitions
WELLS_CTRL = ['A1', 'A2', 'A3', 'A6', 'B1', 'B3', 'B4', 'B5', 'C5', 'D1', 'D2', 'D5']
WELLS_LSD = ['A4', 'A5', 'B2', 'B6', 'C1', 'C2', 'C3', 'C4', 'C6', 'D3', 'D4', 'D6']
DATE_LSD_TREATMENT = "20231115"  # Date of LSD treatment finished. LSD treatment started on 20231114

# Flags
FLAG_PLOT = False

# Folder names for the results
FOLDER_NAME_RASTERPLOTS = "0_1_rasterplots"
FOLDER_NAME_split_data = "1_0_split_data"
FOLDER_NAME_feature_synchrony = "2_0_feature_synchrony"
FOLDER_NAME_feature_set = "3_0_feature_set"


# Parameter which will be calculated
MIN_FR = 0.1 * pq.Hz  # Minimum firing rate in Hz, 0.1 Hz = 6 spikes per minute
BIN_SIZES = [1 * pq.ms] #,10 * pq.ms, 100 * pq.ms] 
WINDOW_SIZES = [120 * pq.s]  # 30 s excluded, took too long.  600 * pq.s excluded, not enough data points for 10 fold cross validation
WINDOW_OVERLAPS = [0.5] # 0.5 = 50% overlap
CONNECTIVITY_METHODS = ["spearman", "canonical", "pearson"]  # tspe excluded, because needs another thresholding method than the other correlation methods
FEATURE_SET_LIST = [
                    FOLDER_NAME_feature_synchrony
    ]

ML_MODELS = ['RF', 'XGboost', 'SVM', 'NB', 'LR', 'KNN', 'MLP']
