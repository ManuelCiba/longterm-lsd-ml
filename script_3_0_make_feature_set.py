import os
from lib.machine_learning import workflow_new
from lib.data_handler import folder_structure
from lib.data_handler import hd
import settings
import numpy as np
import pandas as pd

SOURCE_DATA_FOLDER = settings.FOLDER_NAME_feature_synchrony


def save_dataframe(df_bic00, df_bic10, target_path):
    full_path_bic00 = os.path.join(target_path, 'bic00.csv')
    full_path_bic10 = os.path.join(target_path, 'bic10.csv')

    hd.save_df_as_csv(df_bic00, full_path_bic00)
    hd.save_df_as_csv(df_bic10, full_path_bic10)

def merge_list_of_df_to_one_df(df_list):
    merged_df = pd.concat(df_list, axis=0, ignore_index=True)
    return merged_df

def merge_two_lists_together(list_a, list_b):
    list_final = [np.concatenate((a, b)) for a, b in zip(list_a, list_b)]
    return list_final

def merge_two_dataframes_together(df1, df2):

    merged_df = pd.concat([df1, df2], axis=1)
    return merged_df

if __name__ == '__main__':

    # define parameter
    bin_sizes = settings.BIN_SIZES
    window_sizes = settings.WINDOW_SIZES
    window_overlaps = settings.WINDOW_OVERLAPS
    chip_names = settings.WELLS_CTRL + settings.WELLS_LSD  # Combine control and LSD wells

    path_experiment_list = folder_structure.generate_paths(target_data_folder=SOURCE_DATA_FOLDER,
                                                     bin_sizes=bin_sizes,
                                                     window_sizes=window_sizes,
                                                     window_overlaps=window_overlaps,
                                                     chip_names=[],
                                                     groups=[])


    for path_experiment in path_experiment_list:

        # FEATURES: only synchrony-curve
        source_path = path_experiment
        TARGET_DATA_FOLDER = settings.FOLDER_NAME_feature_set
        target_path = source_path.replace(SOURCE_DATA_FOLDER, TARGET_DATA_FOLDER)

        # List all directories in the current path
        current_path = source_path
        chip_folders = [entry for entry in os.listdir(current_path) if os.path.isdir(os.path.join(current_path, entry))]
        # sort the chip folders
        chip_folders.sort()

        # Initialize a list to hold DataFrames
        df_list = []

        # Loop through each folder (=Chip, e.g. A1) and process the files
        for chip_folder in chip_folders:
            # Define the full path to the folder
            chip_folder_path = os.path.join(current_path, chip_folder)

            
            # Define label LSD or CTRL based on the folder name
            if chip_folder in settings.WELLS_LSD:
                label_group = 'LSD'
                label_y = 1
            elif chip_folder in settings.WELLS_CTRL:
                label_group = 'CTRL'
                label_y = 0
            else:
                raise ValueError(f"Folder {chip_folder} does not match any known labels (LSD or CTRL).")

            # Get all folder of the current path
            rec_folders = [entry for entry in os.listdir(chip_folder_path) if os.path.isdir(os.path.join(chip_folder_path, entry))]
            # sort the recording folders
            rec_folders.sort()

            # Loop through each recording folder
            for rec_folder in rec_folders:
                # Define the full path to the recording folder
                rec_folder_path = os.path.join(chip_folder_path, rec_folder)

                # define label that represents the days after LSD treatment
                # get the date from the folder name
                date_of_file = rec_folder.split('_')[0]
                # calculate the days after LSD treatment
                date_of_treatment = settings.DATE_LSD_TREATMENT
                label_days_after_treatment = (pd.to_datetime(date_of_file) - pd.to_datetime(date_of_treatment)).days


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
                    # Append the DataFrame to the list
                    df_list.append(df)

        # transform df_list to a single DataFrame
        df_feature_set = merge_list_of_df_to_one_df(df_list)
        
        # save ethe DataFrame
        full_target_path = os.path.join(target_path, 'feature_set.csv')
        hd.save_df_as_csv(df_feature_set, full_target_path)

  