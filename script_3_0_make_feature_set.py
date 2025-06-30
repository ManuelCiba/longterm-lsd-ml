import os
from lib.machine_learning import workflow_paired
from lib.data_handler import folder_structure
from lib.data_handler import hd
import settings
import numpy as np
import pandas as pd
from datetime import datetime


def get_feature_set_df(source_path):
    # List all directories in the current path
    current_path = source_path
    chip_folders = [entry for entry in os.listdir(current_path) if os.path.isdir(os.path.join(current_path, entry))]
    # sort the chip folders
    chip_folders.sort()

    # Initialize a list to hold DataFrames
    df_list = []

    # Loop through each folder (=Chip, e.g. A1) and process the files
    for chip_folder in chip_folders:

        print(f"Processing chip folder: {chip_folder}")

        # Define the full path to the folder
        chip_folder_path = os.path.join(current_path, chip_folder)

        
        # Define label DRUG or SHAM based on the folder name
        if chip_folder in settings.WELLS_DRUG:
            label_group = 'DRUG'
            label_y = 1
        elif chip_folder in settings.WELLS_SHAM:
            label_group = 'SHAM'
            label_y = 0
        else:
            print(f"Warning: {chip_folder} is not in the defined groups (DRUG or SHAM). Skipping this folder.")
            continue

        # Get all folder of the current path
        rec_folders = [entry for entry in os.listdir(chip_folder_path) if os.path.isdir(os.path.join(chip_folder_path, entry))]
        # sort the recording folders
        rec_folders.sort()

        # Loop through each recording folder
        for rec_folder in rec_folders:

            print(f"Processing recording folder: {rec_folder}") 

            # Define the full path to the recording folder
            rec_folder_path = os.path.join(chip_folder_path, rec_folder)

            # define label that represents the days after DRUG treatment
            # get the date from the folder name
            date_time_of_file = datetime.strptime(rec_folder, "%Y%m%d_%H%M%S")
            date_time_DRUG_finished = datetime.strptime(settings.DATE_TIME_DRUG_TREATMENT_FINISHED, "%Y%m%d_%H%M%S")
            
            # Calculate the difference as a fraction of a day
            delta = date_time_of_file - date_time_DRUG_finished
            label_days_after_treatment = delta.total_seconds() / (24 * 3600)

            # List all .csv files in the folder
            files = [f for f in os.listdir(rec_folder_path) if f.endswith('.csv')]
            # sort the files
            files.sort()

            # Load each .csv file
            for file in files:
                source_file = os.path.join(rec_folder_path, file)

                # load the DataFrame
                df = hd.load_csv_as_df(source_file, index_col=0)

                # Add a new column for the label
                df['group'] = label_group
                df['y'] = label_y
                df['days_after_treatment'] = label_days_after_treatment
                df['chip'] = chip_folder
                df['recording'] = rec_folder
                df['file'] = file
                # Append the DataFrame to the list
                df_list.append(df)

    # transform df_list to a single DataFrame
    df_feature_set = _merge_list_of_df_to_one_df(df_list)

    return df_feature_set

def _merge_list_of_df_to_one_df(df_list):
    merged_df = pd.concat(df_list, axis=0, ignore_index=True)
    return merged_df

def _merge_two_lists_together(list_a, list_b):
    list_final = [np.concatenate((a, b)) for a, b in zip(list_a, list_b)]
    return list_final

def _merge_two_dataframes_together(df1, df2):
    merged_df = pd.concat([df1, df2], axis=1)
    return merged_df

if __name__ == '__main__':

    # define parameter
    bin_sizes = settings.BIN_SIZES
    window_sizes = settings.WINDOW_SIZES
    window_overlaps = settings.WINDOW_OVERLAPS
    chip_names = settings.WELLS_SHAM + settings.WELLS_DRUG  # Combine control and DRUG wells

    path_experiment_list = folder_structure.generate_paths(target_data_folder="TARGET_DATA_FOLDER",
                                                     bin_sizes=bin_sizes,
                                                     window_sizes=window_sizes,
                                                     window_overlaps=window_overlaps,
                                                     chip_names=[],
                                                     groups=[])


    for path_experiment in path_experiment_list:

        ############################################################
        # FEATURE-SET 1: only synchrony-curve  
        SOURCE_DATA_FOLDER = settings.FOLDER_NAME_feature_synchrony 
        TARGET_DATA_FOLDER = settings.FOLDER_NAME_feature_set_synchrony_curve
        source_path = path_experiment.replace("TARGET_DATA_FOLDER", SOURCE_DATA_FOLDER)
        target_path = path_experiment.replace("TARGET_DATA_FOLDER", TARGET_DATA_FOLDER)
        # get the feature set DataFrame
        df_feature_set = get_feature_set_df(source_path)
        # save the DataFrame
        full_target_path = os.path.join(target_path, 'feature_set_raw.csv')
        hd.save_df_as_csv(df_feature_set, full_target_path)
        print(f"Saved feature set to {full_target_path}")


        ############################################################
        # FEATURE-SET 2: only synchrony stats  
        SOURCE_DATA_FOLDER = settings.FOLDER_NAME_feature_synchrony_stats 
        TARGET_DATA_FOLDER = settings.FOLDER_NAME_feature_set_synchrony_stats
        source_path = path_experiment.replace("TARGET_DATA_FOLDER", SOURCE_DATA_FOLDER)
        target_path = path_experiment.replace("TARGET_DATA_FOLDER", TARGET_DATA_FOLDER)
        # get the feature set DataFrame
        df_feature_set = get_feature_set_df(source_path)
        # save the DataFrame
        full_target_path = os.path.join(target_path, 'feature_set_raw.csv')
        hd.save_df_as_csv(df_feature_set, full_target_path)
        print(f"Saved feature set to {full_target_path}")

        ############################################################
        # FEATURE-SET 3: synchrony-curve and days (not one-hot-encoded) 
        SOURCE_DATA_FOLDER = settings.FOLDER_NAME_feature_synchrony 
        TARGET_DATA_FOLDER = settings.FOLDER_NAME_feature_set_synchrony_curve_days
        source_path = path_experiment.replace("TARGET_DATA_FOLDER", SOURCE_DATA_FOLDER)
        target_path = path_experiment.replace("TARGET_DATA_FOLDER", TARGET_DATA_FOLDER)
        # get the feature set DataFrame
        df_feature_set = get_feature_set_df(source_path)
        # save the DataFrame
        full_target_path = os.path.join(target_path, 'feature_set_raw.csv')
        hd.save_df_as_csv(df_feature_set, full_target_path)
        print(f"Saved feature set to {full_target_path}")


        ############################################################
        # FEATURE-SET 4: synchrony stats and days (not one-hot-encoded) 
        SOURCE_DATA_FOLDER = settings.FOLDER_NAME_feature_synchrony_stats 
        TARGET_DATA_FOLDER = settings.FOLDER_NAME_feature_set_synchrony_stats_days
        source_path = path_experiment.replace("TARGET_DATA_FOLDER", SOURCE_DATA_FOLDER)
        target_path = path_experiment.replace("TARGET_DATA_FOLDER", TARGET_DATA_FOLDER)
        # get the feature set DataFrame
        df_feature_set = get_feature_set_df(source_path)
        # save the DataFrame
        full_target_path = os.path.join(target_path, 'feature_set_raw.csv')
        hd.save_df_as_csv(df_feature_set, full_target_path)
        print(f"Saved feature set to {full_target_path}")

  