import os

from lib.machine_learning import workflow_unpaired
from lib.data_handler import folder_structure
from lib.data_handler import hd
import settings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def bootstrap_ci(data, n_iterations=1000, ci_percentile=95):
    np.random.seed(13)
    # Bootstrapping
    means = []
    for _ in range(n_iterations):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        means.append(np.mean(bootstrap_sample))

    # Calculate the percentiles for the confidence interval
    lower_percentile = (100 - ci_percentile) / 2
    upper_percentile = 100 - lower_percentile
    lower_ci = np.percentile(means, lower_percentile)
    upper_ci = np.percentile(means, upper_percentile)

    return lower_ci, upper_ci

if __name__ == '__main__':

     # for all machine learning folder:
    for FOLDER in settings.FEATURE_SET_LIST:

        print(FOLDER)

        SOURCE_DATA_FOLDER = FOLDER.replace("0_feature_set", "1_ML")

        base_folder = os.path.join(settings.PATH_RESULTS_FOLDER, SOURCE_DATA_FOLDER)

        bin_sizes = settings.BIN_SIZES
        window_sizes = settings.WINDOW_SIZES
        window_overlaps = settings.WINDOW_OVERLAPS
        chip_names = settings.WELLS_SHAM + settings.WELLS_DRUG

        # Create an empty list
        list_results_test = []

        for bin_size in bin_sizes:
            for window_size in window_sizes:
                for window_overlap in window_overlaps:

                    # define path where df_results are saved
                    path_experiment = os.path.join(base_folder,
                                                    f"bin_{bin_size}",
                                                    f"window_{window_size}",
                                                    f"overlap_{window_overlap}")


                    # load ml results
                    df_train_results, df_test_results, X_train_list, y_train_list, X_test_list, y_test_list = workflow_unpaired.load_df_results_from_hd(path_experiment)

                    # Average test results across all splits in order to get a single value to evaluate the model
                    df_results = df_test_results.groupby("Model").mean().reset_index()

                    # Calculate the confidence interval for each metric
                    for metric in ["AUC"]:  # Add your relevant metrics here
                        for model in df_results["Model"].unique():
                            # Filter the results for each model and metric
                            model_data = df_test_results[df_test_results["Model"] == model][metric]

                            # Get the confidence interval
                            lower_ci, upper_ci = bootstrap_ci(model_data)

                            # Add the confidence intervals to the results dataframe
                            df_results.loc[df_results["Model"] == model, f"{metric}_CI_lower"] = lower_ci
                            df_results.loc[df_results["Model"] == model, f"{metric}_CI_upper"] = upper_ci


                    # for each ML model
                    for idx_model in range(len(df_results['Model'])):
                    #    # get model name
                        model = df_results['Model'][idx_model]
                    #    # fill in table
                        list_results_test.append({  'Bin size': int(bin_size),
                                                    'Window size': int(window_size),
                                                    'Window overlap': window_overlap,
                                                    'ML model': model,
                                                    'AUC': df_results.at[idx_model, 'AUC'],
                                                    'AUC_CI_lower': df_results.at[idx_model, 'AUC_CI_lower'],
                                                    'AUC_CI_upper': df_results.at[idx_model, 'AUC_CI_upper'],
                                                    'Accuracy': df_results.at[idx_model, 'Accuracy'],
                                                    'Recall': df_results.at[idx_model, 'Recall'],
                                                    'Precision': df_results.at[idx_model, 'Precision'],
                                                    'F1': df_results.at[idx_model, 'F1'],
                                                    })

        # make DataFrame and sort
        df_results_test = pd.DataFrame(list_results_test)
        df_sorted = df_results_test.sort_values(by='AUC_CI_lower', ascending=False)

        # save dataframe as csv
        full_path = os.path.join(base_folder, "results_test.csv")
        hd.save_df_as_csv(df_sorted, full_path)