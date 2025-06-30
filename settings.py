import os
import quantities as pq

FLAG_DATA_ISMAEL = False    # Set to True if you want to use Ismael's data
FLAG_DATA_WENUS = False    # Set to True if you want to use WENUS data
FLAG_DATA_BDNF = True    # Set to True if you want to use BDNF data

if FLAG_DATA_ISMAEL:
    # local path to the data (laptop)
    #PATH_DATA_FOLDER = "/home/mc/Documents/Data/DRUG/IsmaelLozanoFlores/LongTerm/TS/"

    # KAI path to the data (server)
    PATH_DATA_FOLDER = "/home/mciba/Documents/2025-06_LSD/data/longterm_pot_ismael/TS/"

    # Path where results will be saved
    PATH_RESULTS_FOLDER = os.path.join(os.getcwd(), "results", "Ismael")

    # Ismael's data:
    # DRUG (10µM) and SHAM (1% DMSO) well definitions
    WELLS_SHAM = ['A2', 'A3', 'A6', 'B1', 'B3', 'B4', 'B5', 'C5', 'D1', 'D2', 'D5'] # A1 excluded, because no baseline data available
    WELLS_DRUG = ['A4', 'A5', 'B2', 'B6', 'C1', 'C2', 'C3', 'C4', 'C6', 'D3', 'D4', 'D6']
    DATE_TIME_DRUG_TREATMENT_STARTED = "20231114_143512"  # At this time DRUG was present. 20231114_143512 = 14.11.2023, 14:35:12
    DATE_TIME_DRUG_TREATMENT_FINISHED = "20231115_140043"  # At this time DRUG was already washed out. 20231115_140043 = 15.11.2023, 14:00:43

if FLAG_DATA_WENUS:
    # KAI path to the data (server)
    PATH_DATA_FOLDER = "/home/mciba/Documents/2025-06_LSD/data/longterm_pot_WenusNafez/TS/"

    # Path where results will be saved
    PATH_RESULTS_FOLDER = os.path.join(os.getcwd(), "results", "Wenus")

    # Wenus's data:
    # DRUG (10µM) and SHAM (1% DMSO) well definitions
    WELLS_SHAM = ['B1w', 'C6w', 'D3w', 'D4w', 'D5w', 'D6w']
    WELLS_DRUG = ['C1w', 'C2w', 'C3w', 'C4w', 'C5w', 'D1w', 'D2w'] # maybe exclud: C1, D2
    DATE_TIME_DRUG_TREATMENT_STARTED = "20230726_000000"  # TODO: check the exact time when DRUG was present.
    DATE_TIME_DRUG_TREATMENT_FINISHED = "20230731_000000"  # TODO: check the exact time when DRUG was washed out

if FLAG_DATA_BDNF:
    # KAI path to the data (server)
    PATH_DATA_FOLDER = "/home/mciba/Documents/2025-06_LSD/data/longterm_pot_BDNF/TS/"

    # Path where results will be saved
    PATH_RESULTS_FOLDER = os.path.join(os.getcwd(), "results", "BDNF")

    # BDNF data:
    # Day of Plating: 05.06.2025
    # Application of BDNF: 8 div, 50 ng/ml (dissolved in DH2O), 
    # 13.06.-16.06.2025 (72h)
    WELLS_SHAM = ['B1_Plate1', 'B1_Plate2', 'B1_Plate3', 
                  'B2_Plate1', 'B2_Plate2', 'B2_Plate3',
                  'B3_Plate1', 'B3_Plate2', 'B3_Plate3']
    WELLS_DRUG = ['A1_Plate1', 'A1_Plate2', 'A1_Plate3', 
                 'A2_Plate1', 'A2_Plate2', 'A2_Plate3',
                 'A3_Plate1', 'A3_Plate2', 'A3_Plate3']
    DATE_TIME_DRUG_TREATMENT_STARTED = "20250613_124000"  # TODO: check the exact time when DRUG was present.
    DATE_TIME_DRUG_TREATMENT_FINISHED = "20250616_000000"  # TODO: check the exact time when DRUG was washed out


# Flags to control the analysis
FLAG_PLOT = True
FLAG_LONGTERM = True # Set to True if you want to analyze the long-term data
FLAG_ACUTE = False # Set to True if you want to analyze the acute data

# Folder names for the results
FOLDER_NAME_RASTERPLOTS = "0_1_rasterplots"
FOLDER_NAME_split_data = "1_0_split_data"
FOLDER_NAME_feature_synchrony = "2_0_feature_synchrony"
FOLDER_NAME_feature_synchrony_stats = "2_0_feature_synchrony_stats"
FOLDER_NAME_feature_set_synchrony_curve = "3_0_feature_set_synchrony_curve"
FOLDER_NAME_feature_set_synchrony_stats = "3_0_feature_set_synchrony_stats"
FOLDER_NAME_feature_set_synchrony_curve_days = "3_0_feature_set_synchrony_curve_days"
FOLDER_NAME_feature_set_synchrony_stats_days = "3_0_feature_set_synchrony_stats_days"

# Parameter which will be calculated
MIN_FR = 0.1 * pq.Hz  # Minimum firing rate in Hz, 0.1 Hz = 6 spikes per minute
BIN_SIZES = [1 * pq.ms] #,10 * pq.ms, 100 * pq.ms] 
WINDOW_SIZES = [480 * pq.s]#[60 * pq.s, 120 * pq.s, 240 * pq.s]  # 30 s excluded, took too long.  600 * pq.s excluded, not enough data points for 10 fold cross validation
WINDOW_OVERLAPS = [0.5] # 0.5 = 50% overlap
FEATURE_SET_LIST = [
                    #FOLDER_NAME_feature_set_synchrony_curve,
                    #FOLDER_NAME_feature_set_synchrony_stats,
                    FOLDER_NAME_feature_set_synchrony_curve_days,
                    FOLDER_NAME_feature_set_synchrony_stats_days,
    ]
#ML_MODELS = ['RF', 'SVM', 'NB', 'LR', 'KNN']
ML_MODELS = ['RF', 'XGboost', 'SVM', 'NB', 'LR', 'KNN', 'MLP']
