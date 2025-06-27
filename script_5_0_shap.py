import os
import quantities as pq
from matplotlib.colors import ListedColormap
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from lib.machine_learning import workflow_unpaired
from lib.data_handler import folder_structure
from lib.data_handler import hd
import settings
import pandas as pd
import matplotlib.pyplot as plt
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.cm as cm

def _calculate_feature_change(X_train_list, y_train_list):
    # Initialize a DataFrame to store the mean changes for each feature
    feature_change_df = pd.DataFrame()

    # Loop over each split
    for i in range(len(X_train_list)):
        # Get the feature data and labels for this split
        X_train = X_train_list[i]
        y_train = y_train_list[i]

        # Calculate mean feature values for label 0 (before)
        #before_mean = X_train[y_train.iloc[:, 0] == 0].mean()

        # Calculate mean feature values for label 1 (after)
        #after_mean = X_train[y_train.iloc[:, 0] == 1].mean()

        # If y_train is a pandas Series of 0/1 labels:
        before_mean = X_train[y_train == 0].mean()
        after_mean  = X_train[y_train == 1].mean()

        # Calculate the change (after - before)
        change = after_mean - before_mean

        # Append the changes to the feature_change_df
        feature_change_df[f'Split_{i + 1}'] = change

    # Now, feature_change_df contains the mean changes for each feature across the 9 splits
    # Calculate the overall mean change across splits
    overall_mean_change = feature_change_df.mean(axis=1)

    return overall_mean_change

def _combine_shap_values_across_splits(shap_values_list):
    # Calculate mean SHAP values for each split (use absolute values)
    mean_shap_list = [np.abs(shap_value.values.mean(axis=0)) for shap_value in shap_values_list]

    # check if multi-class
    if mean_shap_list[0].ndim == 2:
        # Only use SHAP values of class 1
        _class = 1
        mean_shap_list = [mean_shap[:, _class] for mean_shap in mean_shap_list]
    elif mean_shap_list[0].ndim == 1:
        # only one class is stored, continue
        pass

    # Create a DataFrame
    shap_values_df = pd.DataFrame(mean_shap_list, columns=X_train_list[0].columns)

    # Calculate median and min/max across all splits
    median_shap_values = shap_values_df.median()
    min_shap_values = shap_values_df.min()
    max_shap_values = shap_values_df.max()

    return median_shap_values, min_shap_values, max_shap_values

def _combined_shap_plot(model_name, shap_values_list, X_train_list, y_train_list):

    # get median and min max values across splits
    median_shap_values, min_shap_values, max_shap_values = _combine_shap_values_across_splits(shap_values_list)

    # calculate feature change (if feature value went up or down)
    change = _calculate_feature_change(X_train_list, y_train_list)

    # Create a DataFrame for plotting
    ranked_shap_df = pd.DataFrame({
        'Median SHAP Value': median_shap_values,
        'Min SHAP Value': min_shap_values,
        'Max SHAP Value': max_shap_values,
        'Change': change  # Keep the original change values to determine the color
    })

    # Sort by median SHAP values in descending order
    ranked_shap_df = ranked_shap_df.sort_values(by='Median SHAP Value', ascending=True)

    # Normalize change values to map them to colormap
    # Use np.clip to limit the change values to a range (optional)
    #norm_change = np.clip(ranked_shap_df['Change'], -1, 1)  # Adjust the range as necessary
    norm_change = ranked_shap_df['Change']
    norm_change = norm_change / np.max(norm_change) # -1 ... +1
    norm_change = (norm_change + 1) / 2  # 0 ... 1

    # Get the colormap
    #cmap = cm.get_cmap('coolwarm', 256)  # Get a colormap with 256 levels
    end_color = '#f70062'  # Red
    middle_color = '#ffffff'  # white
    start_color = '#0187fb'  # Blue
    cmap = LinearSegmentedColormap.from_list('custom_cmap', [start_color, middle_color, end_color], N=256)

    # Map the normalized change to the colormap
    colors = cmap(norm_change)

    # Plot with error bars
    # Create subplots
    fig, ax = plt.subplots(1, 1, figsize=(4, 16))

    bars = plt.barh(ranked_shap_df.index, ranked_shap_df['Median SHAP Value'],
                    xerr=[ranked_shap_df['Median SHAP Value'] - ranked_shap_df['Min SHAP Value'],
                          ranked_shap_df['Max SHAP Value'] - ranked_shap_df['Median SHAP Value']],
                    color=colors,
                    capsize=3  # Adjust this number to change the length of the whiskers
)

    # limit number of x ticks to 3
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=3))

    # Remove the box around the axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    plt.xlabel("Absolute SHAP Value", fontsize=12)
    plt.title("Feature Ranking for " + model_name, fontsize=12)

    # Create color bar for the Viridis colormap
    cbar = plt.colorbar(cm.ScalarMappable(cmap=cmap), ax=plt.gca(), shrink=0.8)
    cbar.set_label('Change in Feature Value', fontsize=12)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['↓', '-', '↑'])
    cbar.outline.set_visible(False)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    return fig

def plot_all_shap_values(base_folder, models, X_train_list, y_train_list):
    # Loop through all models
    for m in range(len(models)):
        model_name = models[m]
        print(model_name)

        # Define filename to load
        full_path = os.path.join(base_folder, "shap_values_list_" + model_name + ".pkl")

        shap_values_list = hd.load_pkl_as_list(full_path)
        fig = _combined_shap_plot(model_name, shap_values_list, X_train_list, y_train_list)

        full_path = os.path.join(base_folder, "SHAP_all_" + model_name + ".jpg")
        hd.save_figure(fig, full_path)

        full_path = os.path.join(base_folder, "SHAP_all_" + model_name + ".pdf")
        hd.save_figure(fig, full_path)

def plot_shap_values(base_folder, split_idx, model_name, shap_values):
    # Plot the SHAP values
    # Create subplots
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    # shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
    # shap.summary_plot(shap_values, X_train, plot_type="bar", class_names=["00bic", "10bic"])

    # if dimension is 3, it is a multiclass format
    if len(shap_values.shape) == 3:
        shap.plots.beeswarm(shap_values[:, :, 1])  # only plot class1
    else:
        shap.plots.beeswarm(shap_values)
    # plt.title(f"SHAP Values for Model: {model_name} (Split {i + 1})")
    full_path = os.path.join(base_folder, "shap_" + model_name + "_split_" + str(split_idx) + ".pdf")
    hd.save_figure(fig, full_path)
    full_path = os.path.join(base_folder, "shap_" + model_name + "_split_" + str(split_idx) + ".jpg")
    hd.save_figure(fig, full_path)

def calculate_shap_values(base_folder, models, X_train_list, df_train_results):

    # Loop through all models
    for m in range(len(models)):

        model_name = models[m]

        # Define filename and skip calculation if already calculated
        full_path = os.path.join(base_folder, "shap_values_list_" + model_name + ".pkl")
        if os.path.isfile(full_path):
            print("SHAP values already calculated: " + full_path)
            continue

        shap_values_list = []

        # Loop through each split
        for i in range(len(X_train_list)):

            # Get the best model for this split
            idx_model = (i * 7) + m
            split_model_row = df_train_results.iloc[idx_model]
            if model_name != split_model_row['Model']:
                print("Wrong Model order -> Cancel calculation")
                return False
            best_model = split_model_row['clf_best']

            # Retrieve the training data for the current split
            X_train = X_train_list[i]

            # Create the explainer (different for each model)
            np.random.seed(42)  # fix random seed to get reproducible results
            if model_name in ["RF", "XGboost"]:
                explainer = shap.TreeExplainer(best_model)
            if model_name in ["SVM", "LR", "KNN", "MLP"]:
                explainer = shap.KernelExplainer(best_model.predict_proba, X_train)
            if model_name in ["NB"]:
                explainer = shap.Explainer(best_model.predict_proba, X_train)

            # Calculate SHAP values
            shap_values = explainer(X_train)
            shap_values_list.append(shap_values)

            plot_shap_values(base_folder, i, model_name, shap_values)

        # save shap_values_list
        hd.save_list_as_pkl(shap_values_list, full_path)


def plot_shap_correlation(base_folder, models):
    shap_model_list = []

    # Loop through all models to load and compute median SHAP values
    for model_name in models:
        print(f"Processing model: {model_name}")

        # Define filename to load
        full_path = os.path.join(base_folder, f"shap_values_list_{model_name}.pkl")
        shap_values_list = hd.load_pkl_as_list(full_path)

        # Combine SHAP values across splits to get median values
        median_shap_values, _, _ = _combine_shap_values_across_splits(shap_values_list)

        # Append the median SHAP values (1D array) for this model
        shap_model_list.append(median_shap_values)

    # Stack the list into a DataFrame to create a model-feature matrix
    shap_df = pd.DataFrame(shap_model_list, index=models).T

    # Calculate the correlation matrix between models based on SHAP values
    correlation_matrix = shap_df.corr()

    # Plot the correlation matrix as a heatmap
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='Spectral', vmin=-1, vmax=1,
                cbar_kws={'label': 'Correlation Coefficient'})
    plt.title("Correlation of SHAP Values Between Models")
    plt.xlabel("Models")
    plt.ylabel("Models")

    full_path = os.path.join(base_folder, "SHAP_correlation.pdf")
    hd.save_figure(fig, full_path)



if __name__ == '__main__':
    # for all machine learning folder:
    for FOLDER in settings.FEATURE_SET_LIST:

        print(FOLDER)

        SOURCE_DATA_FOLDER = FOLDER.replace("0_feature_set", "1_ML")

        base_folder = os.path.join(settings.PATH_RESULTS_FOLDER, SOURCE_DATA_FOLDER)

        # Define parameter set for which SHAP will be calculated
        bin_size = 1 * pq.ms
        window_size = 480 * pq.s
        window_overlap = 0.5
        models = settings.ML_MODELS

        # define path where df_results are saved
        path_experiment = os.path.join(base_folder,
                                     f"bin_{bin_size}",
                                     f"window_{window_size}",
                                     f"overlap_{window_overlap}")


        # load ml results
        df_train_results, df_test_results, X_train_list, y_train_list, X_test_list, y_test_list = workflow_unpaired.load_df_results_from_hd(path_experiment)

        # calculate shap values
        calculate_shap_values(base_folder, models, X_train_list, df_train_results)

        # plot shap all values
        plot_all_shap_values(base_folder, models, X_train_list, y_train_list)

        # plot shap correlation
        plot_shap_correlation(base_folder, models)