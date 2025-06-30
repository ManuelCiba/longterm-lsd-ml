import os
import settings
from lib.data_handler import folder_structure
from lib.data_handler import hd
import pandas as pd
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu



def plot_violin_with_group_stats(df, stat_test_func, y_col="max_synchrony", x_col="days_after_treatment", group_col="group"):
    """
    Creates a violin-box-swarm plot with statistical significance stars for DRUG vs SHAM groups.

    Args:
        df (pd.DataFrame): Input data frame.
        stat_test_func (function): A function that conducts statistical tests and returns a dictionary of p-values.
        y_col (str): The column containing the y-axis values (default: "max_synchrony").
        x_col (str): The column containing the x-axis categories (default: "days_after_treatment").
        group_col (str): The column containing the group labels (default: "group").
    """

    # Create the plot
    plt.figure(figsize=(12, 6))
    sns.violinplot(
        x=x_col, y=y_col, hue=group_col, data=df, split=True, inner=None, palette="muted", alpha=0.6
    )
    sns.boxplot(
        x=x_col, y=y_col, hue=group_col, data=df, width=0.2, showfliers=False, dodge=True
    )
    sns.swarmplot(
        x=x_col, y=y_col, hue=group_col, data=df, dodge=True, color="k", alpha=0.7
    )

    # Remove duplicate legends
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[:2], labels[:2], title=group_col, loc="upper left")

    # Run statistical tests and plot significance stars
    p_values = stat_test_func(df, y_col, x_col, group_col)
    _add_significance_stars_grouped(p_values, df, x_col, y_col)

    plt.title("Max Synchrony over Days After Treatment", fontsize=16)
    plt.ylabel("Max Synchrony", fontsize=14)
    plt.xlabel("Days After Treatment", fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt.gcf()

def _add_significance_stars_grouped(p_values, group_means, x_col, y_col):
    """
    Adds significance stars to the plot based on p-values for grouped data.

    Args:
        p_values (dict): A dictionary with days as keys and p-values as values.
        group_means (pd.DataFrame): The preprocessed mean data frame.
        x_col (str): The column containing the x-axis categories.
        y_col (str): The column containing the y-axis values.
    """
    for day, p_val in p_values.items():
        if p_val < 0.05:
            star = "*" if p_val >= 0.01 else ("**" if p_val >= 0.001 else "***")
            y_max = group_means[group_means[x_col] == day][y_col].max()
            plt.text(day, y_max + 0.05, star, ha="center", color="black", fontsize=12)

def stat_test_func(df, y_col, x_col, group_col):
    """
    Conducts Mann-Whitney U tests comparing DRUG vs SHAM for each day.

    Args:
        df (pd.DataFrame): Input data frame.
        y_col (str): Column name for the y-axis values.
        x_col (str): Column name for the x-axis categories.
        group_col (str): Column name for the group labels.

    Returns:
        dict: Dictionary of p-values with days as keys.
    """
    unique_days = df[x_col].unique()
    p_values = {}

    for day in unique_days:
        drug_group = df[(df[x_col] == day) & (df[group_col] == "DRUG")][y_col]
        sham_group = df[(df[x_col] == day) & (df[group_col] == "SHAM")][y_col]

        if not drug_group.empty and not sham_group.empty:
            _, p_value = mannwhitneyu(drug_group, sham_group, alternative="two-sided")
            p_values[day] = p_value

    return p_values


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

        ############################################################
        # FEATURE-SET 1: only synchrony-curve
        source_path = path_experiment.replace("PATH", settings.FOLDER_NAME_feature_set_synchrony_curve)
        target_path = path_experiment.replace("PATH", settings.FOLDER_NAME_statistics)   
        # load the feature set DataFrame
        full_source_path = os.path.join(source_path, 'feature_set_raw.csv')
        df = hd.load_csv_as_df(full_source_path)
        # statistical test

        # round "days_after_treatment" to one digit 
        df["days_after_treatment"] = df["days_after_treatment"].round(0)

        # Drop rows where days_after_treatment < 0
        df = df.loc[df["days_after_treatment"] >= 0].reset_index(drop=True)

        # Identify columns with "Spike-contrast" in their name
        synchrony_columns = [col for col in df.columns if "Spike-contrast" in col]

        # Calculate the maximum synchrony value for each row
        df["max_synchrony"] = df[synchrony_columns].max(axis=1)

        # remove all Spike-contrast columns.
        columns_to_remove = [col for col in df.columns if "Spike-contrast" in col]
        df = df.drop(columns=columns_to_remove)

        # Step 2: Calculate the mean of `max_synchrony` for each group and day
        grouped_df = (
            df.groupby(["chip", "days_after_treatment", "group"], as_index=False)["max_synchrony"]
            .mean()
            )

        fig = plot_violin_with_group_stats(grouped_df, stat_test_func)

        # save the DataFrame
        full_target_path = os.path.join(target_path, 'Max_synchrony.pdf')
        hd.save_figure(fig, full_target_path)
        print(f"Saved feature set to {full_target_path}")

 