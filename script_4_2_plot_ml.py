import os
from lib.data_handler import hd
import settings
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

def plot_all(df):
    sns.set_palette("flare")  # crest

    # Define the parameters you want to plot and their desired orders
    parameter_orders = {
        "Window size in s": sorted(df["Window size"].unique()),  # Sorting the unique values
        "Window overlap in %": sorted(df["Window overlap"].unique()),  # Sorting the unique values
        "Bin size in ms": sorted(df["Bin size"].unique()),  # Sorting the unique values
    }

    # Rename to new names with units
    column_mapping = {
        'Window size': 'Window size in s',
        'Bin size': 'Bin size in ms',
        'Window overlap': 'Window overlap in %'
    }

    # Rename the columns in the DataFrame
    df_renamed = df.rename(columns=column_mapping)

    ml_models = settings.ML_MODELS

    # Create subplots
    fig, axes = plt.subplots(1, len(parameter_orders), figsize=(18, 6), sharey=True)

    # Plot each parameter in a separate subplot
    for ax, param in zip(axes, parameter_orders.keys()):
        # Ensure the x-axis is sorted by setting the order in the barplot
        # sns.barplot(x=param, y="AUC", hue="ML model", data=df, ax=ax,
        #            order=parameter_orders[param], hue_order=ml_models)
        sns.stripplot(x=param, y="AUC_CI_lower", hue="ML model", data=df_renamed, ax=ax,
                      order=parameter_orders[param], hue_order=ml_models, jitter=True, dodge=True,
                      palette="viridis")

        ax.set_title(f"AUC by {param}")
        ax.set_xlabel(param)
        ax.set_ylabel("AUC (Lower CI)")

        # Set y-axis ticks from 0 to 1 with 0.1 steps
        ax.set_yticks(np.arange(0, 1.1, 0.1))

        legend = ax.legend(loc='lower center', frameon=True)
        legend.get_frame().set_edgecolor('black')  # Set the border color
        legend.get_frame().set_facecolor('white')  # Set the background color
        legend.get_frame().set_linewidth(1.5)

    # Adjust the layout
    plt.tight_layout()
    # plt.show()

    # save figure
    full_path = os.path.join(base_folder, "Results-attachment.pdf")
    hd.save_figure(fig, full_path)


def plot_paper(df):
    # Example filter values
    window_overlap_value = 75
    bin_size_value = 1   # Example window size in ms

    # Filter the DataFrame based on predefined values
    filtered_df = df[
        (df['Window overlap'] == window_overlap_value) &
        (df['Bin size'] == bin_size_value)
        ]

    ml_models = settings.ML_MODELS

    #Calculate error bars
    filtered_df['AUC_error_lower'] = filtered_df['AUC'] - filtered_df['AUC_CI_lower']
    filtered_df['AUC_error_upper'] = filtered_df['AUC_CI_upper'] - filtered_df['AUC']

    # Create subplots
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Extend palette to match number of models
    palette = sns.color_palette('viridis', n_colors=len(ml_models))

    # Plot the stripplot for bin size
    sns.lineplot(x='Window size', y='AUC_CI_lower', hue='ML model', style='ML model', hue_order=ml_models,
        markers=True, dashes=True, data=filtered_df, palette=palette,
                 markersize=20, alpha=0.5, linewidth=2.5
    )

    # # Add error bars manually for each model
    # for model in ml_models:
    #     model_df = filtered_df[filtered_df['ML model'] == model]
    #     plt.errorbar(
    #         model_df['Window size'], model_df['AUC'],
    #         yerr=[model_df['AUC_error_lower'], model_df['AUC_error_upper']],
    #         fmt='none', capsize=4, elinewidth=1.5, color=palette[ml_models.index(model)]
    #     )

    # # Loop through each model to plot individually with confidence intervals
    # marker_list = ['o', '+', '.', 'x', '*', 'v', '^']
    # for model_idx in range(len(ml_models)):
    #     model = ml_models[model_idx]
    #     model_df = filtered_df[filtered_df['ML model'] == model]
    #
    #     # Plot the AUC line for the model
    #     sns.lineplot(
    #         x='Window size',
    #         y='AUC',
    #         data=model_df,
    #         label=model,
    #         marker=marker_list[model_idx],
    #         markersize=20,
    #         linestyle='-',
    #         ax=ax
    #     )
    #
    #     # Add the confidence interval as a shaded area
    #     ax.fill_between(
    #         model_df['Window size'],
    #         model_df['AUC_CI_lower'],
    #         model_df['AUC_CI_upper'],
    #         alpha=0.1
    #     )

    # Add labels and title
    plt.xlabel('Window size in s', fontsize=12)
    plt.ylabel('AUC (Lower CI)', fontsize=12)
    #plt.title(
    #    f'Window overlap = {window_overlap_value} (relative), Bin size = {bin_size_value} ms, Correlation method = {correlation_method_value}',
    #    fontsize=14)

    ax.set_yticks(np.arange(0.5, 1.01, 0.1))
    ax.set_xticks(settings.WINDOW_SIZES)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    legend = ax.legend(loc='best', frameon=True, fontsize=12)


    # Adjust the layout
    plt.tight_layout()
    plt.show()

    # save figure
    full_path = os.path.join(base_folder, "3Results-WindowSizes.pdf")
    fig = plt.gcf()
    hd.save_figure(fig, full_path)

def plot_metrics(df):
    # Example filter values
    window_overlap_value = 75
    bin_size_value = 1  # Example window size in ms

    # Filter the DataFrame
    filtered_df = df[
        (df['Window overlap'] == window_overlap_value) &
        (df['Bin size'] == bin_size_value)
    ]

    # Calculate error bars
    filtered_df['AUC_error_lower'] = filtered_df['AUC'] - filtered_df['AUC_CI_lower']
    filtered_df['AUC_error_upper'] = filtered_df['AUC_CI_upper'] - filtered_df['AUC']

    # Sort or filter to only include unique model entries if needed
    df_bar = filtered_df.groupby('ML model').agg({
        'AUC': 'mean',
        'AUC_error_lower': 'mean',
        'AUC_error_upper': 'mean'
    }).reset_index()

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # sort models by AUC_min
    df_bar['AUC_min'] = df_bar['AUC'] - df_bar['AUC_error_lower']
    df_bar = df_bar.sort_values(by='AUC_min', ascending=False).reset_index(drop=True)


    # Create bar plot with error bars
    sns.barplot(
        x='ML model',
        y='AUC',
        data=df_bar,
        palette='viridis',
        ax=ax,
        errorbar=None,
    )

    # Add manual error bars
    ax.errorbar(
        x=range(len(df_bar)),
        y=df_bar['AUC'],
        yerr=[df_bar['AUC_error_lower'], df_bar['AUC_error_upper']],
        fmt='none',
        c='black',
        capsize=5
    )

    # add AUCmin as a label
    #for i, row in df_bar.iterrows():
    #    auc_min = row['AUC'] - row['AUC_error_lower']
    #    ax.text(i, row['AUC'] + 0.02, f"AUC_CI_lower: {auc_min:.2f}", ha='center', fontsize=10)

    # Add vertical AUC_min labels inside each bar
    for bar, auc_min in zip(ax.patches, df_bar['AUC_min']):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,        # X center of bar
            height * 0.05,                            # Small padding from bottom
            "AUCmin: " + f"{auc_min:.3f}", 
            ha='center', va='bottom',
            rotation=90, fontsize=12, color='white', weight='bold'
        )

    # add horizontal ref line
    ax.axhline(y=0.7, linestyle='--', color='gray', label='Chance Level')

    # Plot styling
    ax.set_ylim(df_bar['AUC'].min() - 0.05, df_bar['AUC'].max() + 0.1)
    ax.set_title("Model Comparison (Mean AUC ± 95% CI)", fontsize=18)
    ax.set_ylabel("AUC Score", fontsize=18)
    ax.set_xlabel("ML Model", fontsize=18)
    ax.tick_params(axis='x', labelrotation=45)
    #ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.set_ylim(0.0, 1.05)
    ax.tick_params(axis='x', labelsize=16) 
    ax.tick_params(axis='y', labelsize=16) 

    plt.tight_layout()

    # save figure
    full_path = os.path.join(base_folder, "3Results-Metrics.pdf")
    fig = plt.gcf()
    hd.save_figure(fig, full_path)

if __name__ == '__main__':
    # for all machine learning folder:
    for FOLDER in settings.FEATURE_SET_LIST:

        #########################################################
        # define paths and load data
        print(FOLDER)

        SOURCE_DATA_FOLDER = FOLDER.replace("0_feature_set", "1_ML")

        base_folder = os.path.join(settings.PATH_RESULTS_FOLDER, SOURCE_DATA_FOLDER)

        # Test results (final AUC values)
        full_path = os.path.join(base_folder, 'results_test.csv')
        df = hd.load_csv_as_df(full_path)
        # relative to %
        df["Window overlap"] = df["Window overlap"] * 100
        #plot_all(df)
        #plot_paper(df)
        plot_metrics(df)



