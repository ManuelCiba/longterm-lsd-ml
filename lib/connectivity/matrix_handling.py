import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np


def plot_correlation_matrix(correlation_matrix, full_path):
    # create directory
    result_folder = os.path.dirname(full_path)
    os.makedirs(result_folder, exist_ok=True)

    # Create a heatmap using seaborn
    sns.set_theme()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm',
                xticklabels=range(correlation_matrix.shape[1]), yticklabels=range(correlation_matrix.shape[1]),
                vmin=-1, vmax=1)
    #plt.title(correlation_name)
    plt.title(np.nanmean(correlation_matrix))
    plt.xlabel('Channel')
    plt.ylabel('Channel')
    #plt.show()

    plt.savefig(full_path)

    print("    Saved: " + full_path)


def save_correlation_matrix(correlation_matrix, full_path):
    # create directory
    result_folder = os.path.dirname(full_path)
    os.makedirs(result_folder, exist_ok=True)

    # convert np matrix to dataframe with row and colunn names
    names = ["EL" + "{:02d}".format(i) for i in range(correlation_matrix.shape[0])]
    df = pd.DataFrame(data=correlation_matrix, index=names, columns=names)
    df.to_csv(full_path, sep=',')

def load_correlation_matrix(full_path, index_col=0):
    # header=0 and index_col=0 will use the row and col names like in the csv file
    df_matrix = pd.read_csv(full_path, header=0, index_col=index_col)
    return df_matrix

def flatten_df_matrix(df):
    # Flatten the DataFrame while combining row and column names as the column names
    flattened_data = {f"{col}_{row}": value for col in df.columns for row, value in df[col].items()}

    # Create a new DataFrame from the flattened data
    flattened_df = pd.DataFrame([flattened_data])

    return flattened_df