import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import numpy as np
import os
import settings
from lib.data_handler import folder_structure
from lib.data_handler import hd
from datetime import datetime
from scipy.stats import mannwhitneyu



def preprocess_and_plot(df, target_path):
    """
    Preprocess the data and create violin plots with statistical significance testing.
    Args:
        df (pd.DataFrame): Input DataFrame containing the features and metadata.
    """
    # Identify feature columns (exclude non-feature columns)
    non_feature_cols = ["chip", "y", "group", "days_after_treatment", "file", "recording"]
    feature_cols = [col for col in df.columns if col not in non_feature_cols]
    
    # Group by 'chip', 'group', and 'days_after_treatment', calculate the mean
    df_mean = df.groupby(["chip", "group", "days_after_treatment"], as_index=False)[feature_cols].mean()

    # Group by 'group' and 'days_after_treatment' for statistical tests and plotting
    df_grouped = df_mean.groupby(["group", "days_after_treatment"])
    
    # Initialize the Bonferroni correction factor
    num_tests = len(feature_cols) * df_mean["days_after_treatment"].nunique()
    
    for feature in feature_cols:
        # Perform statistical testing
        p_values = {}
        for day in df_mean["days_after_treatment"].unique():
            drug_vals = df_mean[(df_mean["days_after_treatment"] == day) & (df_mean["group"] == "DRUG")][feature]
            sham_vals = df_mean[(df_mean["days_after_treatment"] == day) & (df_mean["group"] == "SHAM")][feature]
            
            # Perform t-test
            if len(drug_vals) > 1 and len(sham_vals) > 1:
                _, p_val = mannwhitneyu(sham_vals, drug_vals, alternative="two-sided")
                p_values[day] = min(p_val * num_tests, 1.0)  # Apply Bonferroni correction
            
        # Create the violin plot
        plt.figure(figsize=(12, 6))
        sns.violinplot(
            x="days_after_treatment", y=feature, hue="group", 
            data=df_mean, split=True, inner="box", palette="muted"
        )
        sns.swarmplot(
            x="days_after_treatment", y=feature, hue="group", 
            data=df_mean, dodge=True, color="k", alpha=0.5, size=5
        )
        plt.title(f"Feature: {feature}")
        plt.xlabel("Days After Treatment")
        plt.ylabel(f"{feature}")
        plt.legend(title="Group", loc="upper right")

        # Add significance stars
        _add_significance_stars_grouped(p_values, df_mean, "days_after_treatment", feature)

        # Save the plot
        plt.tight_layout()
        full_target_path = os.path.join(target_path, feature + '.pdf')
        hd.save_figure(plt.gcf(), full_target_path)
        print(f"Saved feature set to {full_target_path}")
        

def _add_significance_stars_grouped(p_values, group_means, x_col, y_col):
    """
    Adds significance stars to the plot based on p-values for grouped data.

    Args:
        p_values (dict): A dictionary with days as keys and corrected p-values as values.
        group_means (pd.DataFrame): The preprocessed mean data frame.
        x_col (str): The column containing the x-axis categories.
        y_col (str): The column containing the y-axis values.
    """
    for day, p_val in p_values.items():
        if p_val < 0.05:
            star = "*" if p_val >= 0.01 else ("**" if p_val >= 0.001 else "***")
            y_max = group_means[group_means[x_col] == day][y_col].max()
            plt.text(
                x=day, y=y_max + 0.05, s=star, ha="center", color="black", fontsize=12
            )


if __name__ == '__main__':

    # define parameter
    bin_sizes = settings.BIN_SIZES
    window_sizes = settings.WINDOW_SIZES
    window_overlaps = settings.WINDOW_OVERLAPS
    chip_names = settings.WELLS_SHAM + settings.WELLS_DRUG  # Combine control and DRUG wells

    path_experiment_list = folder_structure.generate_paths(target_data_folder="PATH",
                                                     bin_sizes=bin_sizes,
                                                     window_sizes=window_sizes,
                                                     window_overlaps=window_overlaps,
                                                     chip_names=[],
                                                     groups=[])


    for path_experiment in path_experiment_list:

        for feature_set in settings.FEATURE_SET_LIST:

            source_path = path_experiment.replace("PATH", feature_set)
            target_path = path_experiment.replace("PATH", os.path.join(settings.FOLDER_NAME_statistics, feature_set))   
            # load the feature set DataFrame
            full_source_path = os.path.join(source_path, 'feature_set_raw.csv')
            df = hd.load_csv_as_df(full_source_path)
            # statistical test

            # round "days_after_treatment" to one digit 
            df["days_after_treatment"] = df["days_after_treatment"].round(0)

            # Drop rows where days_after_treatment < 0
            df = df.loc[df["days_after_treatment"] >= 0].reset_index(drop=True)

            preprocess_and_plot(df, target_path)

    