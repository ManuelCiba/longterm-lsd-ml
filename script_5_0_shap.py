import os
import quantities as pq
import matplotlib
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

def _combine_shap_explanations(shap_values_list, class_index=1):
    """
    Combine SHAP values from multiple Explanation objects manually.
    """
    all_values = []
    all_base_values = []
    all_data = []

    for expl in shap_values_list:
        values = expl.values
        if values.ndim == 3:  # multiclass
            values = values[:, :, class_index]
        all_values.append(values)
        all_base_values.append(expl.base_values)
        all_data.append(expl.data)

    combined = shap.Explanation(
        values=np.concatenate(all_values, axis=0),
        base_values=np.concatenate(all_base_values, axis=0),
        data=np.concatenate(all_data, axis=0),
        feature_names=shap_values_list[0].feature_names
    )
    return combined

def _combine_all_shap_values(shap_values_list, class_index=1):
    """
    Combine SHAP values across splits and return full matrix of per-sample SHAP values.
    """
    clean_shap_values_list = []
    for shap_values in shap_values_list:
        values = shap_values.values
        if values.ndim == 3:  # multi-class
            values = values[:, :, class_index]
        clean_shap_values_list.append(values)

    combined_shap = np.concatenate(clean_shap_values_list, axis=0)
    return combined_shap

def OLD_calculate_feature_change(X_train_list, y_train_list):
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

    combined_shap = _combine_all_shap_values(shap_values_list)

    # Calculate mean over all samples per split
    mean_abs_shap = np.abs(combined_shap).mean(axis=0)

    # Create a DataFrame
    shap_values_df = pd.DataFrame(mean_abs_shap, columns=X_train_list[0].columns)

    # Calculate median and min/max across all splits
    median_shap_values = shap_values_df.median()
    min_shap_values = shap_values_df.min()
    max_shap_values = shap_values_df.max()

    return median_shap_values, min_shap_values, max_shap_values

def OLD_combined_shap_plot(model_name, shap_values_list, X_train_list, y_train_list):

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
        combined_shap = _combine_shap_explanations(shap_values_list, class_index=1)

        output_path = os.path.join(base_folder, "SHAP_all_" + model_name + ".jpg")
        _plot_shap_values(combined_shap, output_path)
        
        #fig = _combined_shap_plot(model_name, shap_values_list, X_train_list, y_train_list)
        #full_path = os.path.join(base_folder, "SHAP_all_" + model_name + ".jpg")
        #hd.save_figure(fig, full_path)
        #full_path = os.path.join(base_folder, "SHAP_all_" + model_name + ".pdf")
        #hd.save_figure(fig, full_path)

def _plot_shap_values(shap_values, output_path):
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

    hd.save_figure(fig, output_path)
    hd.save_figure(fig, output_path.replace("jpg", "pdf"))

def OLDplot_shap_correlation(base_folder, models):
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

def _plot_shap_dependence(shap_values, X_train, feature_name, output_path, interaction_index=None):
    """
    Generate a SHAP dependence plot for a specific feature.

    Args:
        shap_values: Array-like SHAP values for the dataset.
        X_train: DataFrame of training features.
        feature_name: The name of the feature to plot.
        output_path: Path to save the dependence plot.
        interaction_index: Optional, name of the feature to show interaction effects.
    """
    # Multi-class case: pick SHAP values for the positive class (e.g., class 1)
    if len(shap_values.shape) == 3:
        if shap_values.shape[2] == 2:  # Multi-class SHAP values
            shap_values = shap_values[:, :, 1]  # Use SHAP values for class 1

    # Calculate the mean absolute SHAP values for each feature
    mean_abs_shap = np.median(np.abs(shap_values.values), axis=0)
    
    # Get the most important feature
    most_important_feature = X_train.columns[np.argmax(mean_abs_shap)]
    print(f"Most important feature: {most_important_feature}")

    # Rescale the feature values in X_train to be in unit "days" instead of 0...1
    max_day = 9 # hard coded here for simplicity
    min_day = 2
    X_train_rescaled = X_train.copy()
    X_train_rescaled[feature_name] = min_day + X_train[feature_name] * (max_day - min_day)

    #fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    shap.dependence_plot(
        feature_name,
        shap_values.values,
        X_train_rescaled,
        interaction_index=most_important_feature,
        show=False  # Suppress showing the plot in interactive environments
    )

    hd.save_figure(plt.gcf(), output_path)
    hd.save_figure(plt.gcf(), output_path.replace("jpg", "pdf"))

def _plot_combined_shap_dependence(shap_values_list, X_train_list):
    
    # define which feature should be on the x axis
    feature_name = "days_after_treatment"

    combined_shap = _combine_all_shap_values(shap_values_list, class_index=1)
    combined_X = pd.concat(X_train_list, axis=0)

    # Calculate the median absolute SHAP values for each feature
    median_abs_shap = np.median(np.abs(combined_shap), axis=0)
    
    # Get the most important feature
    most_important_feature = combined_X.columns[np.argmax(median_abs_shap)]
    print(f"Most important feature: {most_important_feature}")

    # Rescale the feature values in X_train to be in unit "days" instead of 0...1
    max_day = 9 # hard coded here for simplicity
    min_day = 2
    combined_X[feature_name] = min_day + combined_X[feature_name] * (max_day - min_day)

    # TODO: DELETE THIS LINE
    most_important_feature = "Spike-contrast (7.753 s)"
    
    shap.dependence_plot(
        "days_after_treatment",
        combined_shap,
        combined_X,
        interaction_index=most_important_feature,
        show=False  # Suppress showing the plot in interactive environments
    )

    return plt.gcf()

def plot_all_shap_dependence(base_folder, models, X_train_list):
    # Loop through all models
    for m in range(len(models)):
        model_name = models[m]
        print(model_name)

        # Define filename to load
        full_path = os.path.join(base_folder, "shap_values_list_" + model_name + ".pkl")

        shap_values_list = hd.load_pkl_as_list(full_path)
        fig = _plot_combined_shap_dependence(shap_values_list, X_train_list)

        full_path = os.path.join(base_folder, "SHAP_all_dependence_" + model_name + ".jpg")
        hd.save_figure(fig, full_path)

        full_path = os.path.join(base_folder, "SHAP_all_dependence_" + model_name + ".pdf")
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
                # "hclustering" to deal better with correlated features
                # Summarize the data to 50 representative samples -> faster calculation
                background = shap.sample(X_train, 50) 
                #masker = shap.maskers._tabular.Tabular(background, clustering="correlation") 
                #explainer = shap.Explainer(best_model, masker=masker)
                explainer = shap.TreeExplainer(best_model, data=background, feature_perturbation="interventional")
            if model_name in ["SVM", "LR", "KNN", "MLP"]:
                background = shap.kmeans(X_train, 50)
                explainer = shap.KernelExplainer(best_model.predict_proba, background)
            if model_name in ["NB"]:
                background = shap.sample(X_train, 50)
                explainer = shap.Explainer(best_model.predict_proba, background)

            # Calculate SHAP values
            X_train_sampled = X_train.sample(20, random_state=42)  # speed up calculation
            shap_values = explainer(X_train_sampled)
            shap_values_list.append(shap_values)

            # plot shap values (classical feature ranking)
            output_path = os.path.join(base_folder, "shap_" + model_name + "_split_" + str(i) + ".jpg")
            _plot_shap_values(shap_values, output_path=output_path)

            # Plot SHAP dependence for `days_after_treatment`
            output_path = os.path.join(base_folder, "shap_dependence_split_" + model_name + str(i) + ".jpg")
            try:
                _plot_shap_dependence(
                    shap_values,
                    X_train,
                    feature_name="days_after_treatment",
                    output_path=output_path,
                    interaction_index=None  # Replace with another feature for interaction if needed
                )
            except Exception as e: 
                print(e)

        # save shap_values_list
        hd.save_list_as_pkl(shap_values_list, full_path)

if __name__ == '__main__':
    # for all machine learning folder:
    for FOLDER in settings.FEATURE_SET_LIST:

        print(FOLDER)

        SOURCE_DATA_FOLDER = FOLDER.replace("0_feature_set", "1_ML")
        TARGET_DATA_FOLDER = FOLDER.replace("0_feature_set", "2_SHAP")

        source_folder = os.path.join(settings.PATH_RESULTS_FOLDER, SOURCE_DATA_FOLDER)
        target_folder = os.path.join(settings.PATH_RESULTS_FOLDER, TARGET_DATA_FOLDER)

        # Define parameter set for which SHAP will be calculated
        bin_size = 1 * pq.ms
        window_size = 240 * pq.s
        window_overlap = 0.75
        models = settings.ML_MODELS

        # define path where df_results are stored
        path_experiment = os.path.join(source_folder,
                                     f"bin_{bin_size}",
                                     f"window_{window_size}",
                                     f"overlap_{window_overlap}")

        target_folder_full = path_experiment.replace(SOURCE_DATA_FOLDER, TARGET_DATA_FOLDER)

        # load ml results
        df_train_results, df_test_results, X_train_list, y_train_list, X_test_list, y_test_list = workflow_unpaired.load_df_results_from_hd(path_experiment)

        # IMPORTANT: use x_train or x_test for shap calculation? 
        # actually use x_test (=unseen data) since here we use ML as an statistical test.
        # But in x_train are more data points (=more stable results) and it does not differ too much from x_train shap results
        # since the cross-validaiton already prevented overfitting

        # calculate shap values
        calculate_shap_values(target_folder_full, models, X_train_list, df_train_results)

        # plot all shap value dependence
        try:
            plot_all_shap_dependence(target_folder_full, models, X_train_list)
        except Exception as e:
            print(e)

        # plot shap all values
        try:
            plot_all_shap_values(target_folder_full, models, X_train_list, y_train_list)
        except Exception as e:
            print(e)

        # plot shap correlation
        #plot_shap_correlation(target_folder, models)