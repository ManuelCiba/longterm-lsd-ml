import warnings
warnings.filterwarnings("ignore")
from sklearn.exceptions import ConvergenceWarning
from joblib import Parallel, delayed
import multiprocessing
import os
import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt


from lib.machine_learning import workflow_unpaired
from lib.data_handler import folder_structure
from lib.data_handler import hd
import settings

def preprocess_df_features_longterm(df_features):
    # calculate the difference between features after DRUG application and before DRUG application
    # this is done by subtracting the mean of the features before DRUG application from the raw features after DRUG application
    
    # `settings.DATE_TIME_DRUG_TREATMENT_START` contains the start date/time as a string
    DATE_TIME_DRUG_TREATMENT_STARTED = pd.to_datetime(settings.DATE_TIME_DRUG_TREATMENT_STARTED, format="%Y%m%d_%H%M%S")
    DATE_TIME_DRUG_TREATMENT_FINISHED = pd.to_datetime(settings.DATE_TIME_DRUG_TREATMENT_FINISHED, format="%Y%m%d_%H%M%S")
    
    # Ensure the 'recording' column is in datetime format
    df_features['recording'] = pd.to_datetime(df_features['recording'], format="%Y%m%d_%H%M%S")

    # If the baseline correction is needed, set to True
    if False:
        # drop the daytime column and insert later again after the calculation
        days_after_treatment = df_features.pop('days_after_treatment')

        # Filter rows before DRUG application
        baseline_rows = df_features[df_features['recording'] < DATE_TIME_DRUG_TREATMENT_STARTED]

        # Select columns containing "Spike-contrast" or "s_"
        spike_contrast_columns = [col for col in df_features.columns if "Spike-contrast" in col or "s_" in col]

        # Calculate the mean baseline curve (mean of all rows before DRUG application)
        mean_baseline_curve = baseline_rows[spike_contrast_columns].mean()

        # Subtract the baseline from all rows after the DRUG application
        df_features.loc[df_features['recording'] >= DATE_TIME_DRUG_TREATMENT_STARTED, spike_contrast_columns] -= mean_baseline_curve

        # Reinsert the days_after_treatment column
        df_features = pd.concat([days_after_treatment, df_features], axis=1)

    # drop the baseline rows to only keep post-treatment data
    df_features = df_features[df_features['recording'] >= DATE_TIME_DRUG_TREATMENT_FINISHED]

    # drop columns that are not needed for the machine learning
    df_features = df_features.drop(columns=['group', 'recording', 'file']) 
    # drop colums that contain any NaN values
    df_features = df_features.dropna(axis=1) 

    return df_features

def preprocess_df_features_acute(df_features):
    # calculate the difference between features after DRUG application and before DRUG application
    # this is done by subtracting the mean of the features before DRUG application from the raw features after DRUG application
    
    # `settings.DATE_TIME_DRUG_TREATMENT_START` contains the start date/time as a string
    DATE_TIME_DRUG_TREATMENT_STARTED = pd.to_datetime(settings.DATE_TIME_DRUG_TREATMENT_STARTED, format="%Y%m%d_%H%M%S")
    DATE_TIME_DRUG_TREATMENT_FINISHED = pd.to_datetime(settings.DATE_TIME_DRUG_TREATMENT_FINISHED, format="%Y%m%d_%H%M%S")
    
    # Ensure the 'recording' column is in datetime format
    df_features['recording'] = pd.to_datetime(df_features['recording'], format="%Y%m%d_%H%M%S")

    # drop the daytime column and insert later again after the calculation
    days_after_treatment = df_features.pop('days_after_treatment')

    # Filter rows before DRUG application
    baseline_rows = df_features[df_features['recording'] < DATE_TIME_DRUG_TREATMENT_STARTED]

    # Select columns containing "Spike-contrast" or "s_"
    spike_contrast_columns = [col for col in df_features.columns if "Spike-contrast" in col or "s_" in col]

    # Calculate the mean baseline curve (mean of all rows before DRUG application)
    mean_baseline_curve = baseline_rows[spike_contrast_columns].mean()

    # Subtract the baseline from all rows after the DRUG application
    df_features.loc[df_features['recording'] >= DATE_TIME_DRUG_TREATMENT_STARTED, spike_contrast_columns] -= mean_baseline_curve

    # Reinsert the days_after_treatment column
    df_features = pd.concat([days_after_treatment, df_features], axis=1)

    # only keep acute data
    df_features = df_features[df_features['recording'] <= DATE_TIME_DRUG_TREATMENT_FINISHED]
    df_features = df_features[df_features['recording'] >= DATE_TIME_DRUG_TREATMENT_STARTED]

    # drop columns that are not needed for the machine learning
    df_features = df_features.drop(columns=['group', 'recording', 'file']) 
    # drop colums that contain any NaN values
    df_features = df_features.dropna(axis=1) 

    # Drop columns where all values are zeros
    df_features = df_features.loc[:, ~(df_features == 0).all()]

    return df_features


def withinloop(path_experiment, SOURCE_DATA_FOLDER, TARGET_DATA_FOLDER):
    source_path = path_experiment

    # check if already calculated
    target_path = source_path.replace(SOURCE_DATA_FOLDER, TARGET_DATA_FOLDER)
    full_path = os.path.join(target_path, 'df_results.pkl')
    if os.path.isfile(full_path):
        print("Already calculated: " + full_path)
        return
    else:
        print("Calculating: " + full_path)

    # define source path
    full_path = os.path.join(path_experiment, 'feature_set.csv')

    # load the feature dataframes
    df_features = hd.load_csv_as_df(full_path, index_col=False)

    if settings.FLAG_LONGTERM:

        # preprocess the feature dataframe
        df_features_longterm = preprocess_df_features_longterm(df_features)

        # perform ml
        df_train_results, df_test_results, X_train_list, y_train_list, X_test_list, y_test_list = workflow_unpaired.call(models, df_features_longterm)
        workflow_unpaired.save_df_results_to_hd(target_path,
                                        df_train_results,
                                        df_test_results,
                                        X_train_list,
                                        y_train_list,
                                        X_test_list,
                                        y_test_list)

        # save figs
        fig_test_path = os.path.join(target_path, "fig_test.pdf")
        fig_train_path = os.path.join(target_path, "fig_train.pdf")
        fig_test = workflow_unpaired.plot_test_results(df_test_results)
        hd.save_figure(fig_test, fig_test_path)
        #hd.save_figure(fig_train, fig_train_path)


    if settings.FLAG_ACUTE:
        # preprocess the feature dataframe
        df_features_acute = preprocess_df_features_acute(df_features)

        # perform ml
        df_train_results, df_test_results, X_train_list, y_train_list, X_test_list, y_test_list = workflow_unpaired.call(models, df_features_acute)
        workflow_unpaired.save_df_results_to_hd(target_path,
                                        df_train_results,
                                        df_test_results,
                                        X_train_list,
                                        y_train_list,
                                        X_test_list,
                                        y_test_list)

        # save figs
        fig_test_path = os.path.join(target_path, "fig_test_acute.pdf")
        fig_train_path = os.path.join(target_path, "fig_train_acute.pdf")
        fig_test = workflow_unpaired.plot_test_results(df_test_results)
        hd.save_figure(fig_test, fig_test_path)
        #hd.save_figure(fig_train, fig_train_path)

    print("Calculation finished: " + full_path)

def _get_path_experiment_list():
    # define parameter
    bin_sizes = settings.BIN_SIZES
    window_sizes = settings.WINDOW_SIZES
    window_overlaps = settings.WINDOW_OVERLAPS

    path_experiment_list = folder_structure.generate_paths(target_data_folder=SOURCE_DATA_FOLDER,
                                                           bin_sizes=bin_sizes,
                                                           window_sizes=window_sizes,
                                                           window_overlaps=window_overlaps,
                                                           chip_names=[],
                                                           groups=[])

    return path_experiment_list

def run_parallized(models, SOURCE_DATA_FOLDER, TARGET_DATA_FOLDER):

    # get all paths of the different parameter combinations
    path_experiment_list = _get_path_experiment_list()

    # define number of CPU cores
    num_cores = int(multiprocessing.cpu_count()/2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=Warning)
        Parallel(n_jobs=num_cores)(delayed(withinloop)(param, SOURCE_DATA_FOLDER, TARGET_DATA_FOLDER) for param in path_experiment_list)

def run(models, SOURCE_DATA_FOLDER, TARGET_DATA_FOLDER):
    
    path_experiment_list = _get_path_experiment_list()


    for path_experiment in path_experiment_list:

        source_path = path_experiment

        # check if already calculated
        target_path = source_path.replace(SOURCE_DATA_FOLDER, TARGET_DATA_FOLDER)
        full_path = os.path.join(target_path, 'df_results.pkl')
        if os.path.isfile(full_path):
            print("Already calculated: " + full_path)
            continue
        else:
            print("Calculating: " + full_path)

        withinloop(path_experiment, SOURCE_DATA_FOLDER, TARGET_DATA_FOLDER)



if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    for SOURCE_DATA_FOLDER in settings.FEATURE_SET_LIST:

        # define folder name to save results
        TARGET_DATA_FOLDER = SOURCE_DATA_FOLDER.replace("0_feature_set", "1_ML")

        # do the machine learning
        models = settings.ML_MODELS
        run_parallized(models, SOURCE_DATA_FOLDER, TARGET_DATA_FOLDER)
        #run(models, SOURCE_DATA_FOLDER, TARGET_DATA_FOLDER)