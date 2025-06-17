import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats  # statistics
from pathlib import Path
import numpy as np
import os

def make_box_plot_for_two_groups(df_group1, df_group2, group_name1, group_name2, feature, xlabel):

    print("Plotting: Boxplot for " + feature)

    # Set global font size
    plt.rcParams.update({'font.size': 14})

    # Add a 'Group' column
    df_group1[xlabel] = group_name1
    df_group2[xlabel] = group_name2

    # Combine the dataframes for the current feature
    df = pd.concat([df_group1, df_group2], ignore_index=True)

    # Create a boxplot for the current feature
    plt.figure(figsize=(6, 4))
    plt.clf()  # Clear the current figure
    sns.boxplot(x=xlabel, y=feature, data=df, hue=xlabel, palette='viridis')
    sns.stripplot(x=xlabel, y=feature, hue=xlabel, data=df, palette='rocket', dodge=False, linewidth=0,
                  alpha=0.7)

    # Perform mann whitney u test
    y1 = df_group1[feature].to_numpy()
    y2 = df_group2[feature].to_numpy()

    try:
        #statistic, p_value = stats.mannwhitneyu(y1, y2)
        statistic, p_value = stats.wilcoxon(y1, y2, nan_policy='omit')

    except:
        p_value = np.nan

    # Add significance annotations manually
    x = 0
    y = df[feature].max()
    _plot_stars(p_value, x, y)

    plt.xticks(rotation=45)

    return plt.gcf()


def _plot_stars(p_value, x, y):
    x1, x2 = x, x+1  # begin and end of horizontal bar (first column: 0, see plt.xticks())
    y = y * 1.1  # y position of horizontal bar and stars
    h = y * 0.01  # height of vertical lines on horizontal bar
    color = 'k'

    symbol = 'ns'
    if p_value < 0.05:
        symbol = '*'
    if p_value < 0.01:
        symbol = '**'
    if p_value < 0.001:
        symbol = '***'

    plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=color)
    plt.text((x1 + x2) * .5, y + h, symbol, ha='center', va='bottom', color=color)