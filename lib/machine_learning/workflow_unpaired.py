import glob
from multiprocessing import Pool
import multiprocessing
import os
import warnings
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics
from numpy import matrix
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score, recall_score, precision_score, f1_score, \
    classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import GroupKFold, GridSearchCV, StratifiedGroupKFold
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from xgboost import XGBClassifier

from lib.connectivity import matrix_handling


def call(models, df_features):
    warnings.filterwarnings("ignore")

    # print number of rows in df_features
    print("Number of rows in df_features: " + str(len(df_features)))
    # remove rows with NaN values
    df_features = df_features.dropna()
    # print number of rows in df_features
    print("Number of rows after removing rows with NaN values: " + str(len(df_features)))

    # split the df_features into y and X and groups
    y = df_features['y']  # 'y' is the column with class labels
    X = df_features.drop(columns=['y', 'chip'])  # Drop 'y' and 'chip_id' columns to get features
    groups = df_features['chip'].values

    # scale the data in X with minmax scaling
    #X_scaled = preprocessing.MinMaxScaler().fit_transform(X)
    #X_scaled = preprocessing.StandardScaler().fit_transform(X)
    #X_scaled = pd.DataFrame(X_scaled, columns=X.columns)  # Convert back to DataFrame with original column names
    #X = X_scaled

    # Remove columns where all values are zero
    X = X.loc[:, (X != 0).any(axis=0)]

    # Step 3: Outer Cross-Validation using Leave-One-Group-Out
    cv = StratifiedGroupKFold(n_splits=5)
    train_result_list = []  # To store the training metrics (inner CV results)
    test_result_list = []   # To store the test set performance (outer test results)
    X_train_list = [] # to store X_train for all splits, needed for SHAP
    y_train_list = []  # to store y_train for all splits, needed for SHAP
    X_test_list = []
    y_test_list = []

    cnt = 0
    for train_idx, test_idx in cv.split(X, y, groups):

        print("Split #" + str(cnt))
        cnt += 1

        # Split the data into training and testing for this fold
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train, groups_test = groups[train_idx], groups[test_idx]
        X_train_list.append(X_train)
        y_train_list.append(y_train)
        X_test_list.append(X_test)
        y_test_list.append(y_test)

        # Step 4: Call the model evaluation process with the outer training data (inner CV results)
        train_result = _calling_the_models(models, X_train, y_train, groups_train)

        # Step 5: Loop through each model in train_result to evaluate them on the outer test set
        for idx, row in train_result.iterrows():
            model_name = row['Model']
            clf_best = row['clf_best']  # Retrieve the best classifier for this model

            # Predict on the outer test set
            y_test_pred = clf_best.predict(X_test)
            auc_test = roc_auc_score(y_test, y_test_pred)
            accuracy_test = accuracy_score(y_test, y_test_pred)
            recall_test = recall_score(y_test, y_test_pred)
            precision_test = precision_score(y_test, y_test_pred, zero_division=0)
            f1_test = f1_score(y_test, y_test_pred)

            # Collect test set metrics for this model
            test_result = {
                'Model': model_name,
                'AUC': auc_test,
                'Accuracy': accuracy_test,
                'Recall': recall_test,
                'Precision': precision_test,
                'F1': f1_test
            }

            # Append the current model's test results (for this outer fold) to the list
            test_result_list.append(test_result)

        # Append the inner cross-validation results (train) for this outer fold
        train_result_list.append(train_result)

    # Step 6: Combine the results across all outer folds
    df_train_results = pd.concat(train_result_list, ignore_index=True)  # Combine inner CV results into a DataFrame
    df_test_results = pd.DataFrame(test_result_list)  # Combine outer test results into a DataFrame

    return df_train_results, df_test_results, X_train_list, y_train_list, X_test_list, y_test_list


def _calling_the_models(models, X_train, y_train, groups_train):
    # Convert into a 1D array (Important for some ML models)
    y_train = y_train.values.ravel()

    # Initialize to an empty list
    train_results = []
    try:
        with Pool(processes=len(models)) as pool:
            train_results = pool.starmap(_parallel_evaluate,
                                         [(model, X_train, y_train, groups_train) for model in models])
    except Exception as e:
        print(f"An error occurred during model evaluation: {e}")
        return None  # Return None or an empty list to indicate failure

    if not train_results:  # Check if results are empty
        print("No results were returned from the model evaluations.")
        return None  # or return []

    # Convert list to DataFrame; ensure the list is not empty
    try:
        df_train_results = pd.DataFrame(train_results)
    except Exception as e:
        print(f"Failed to convert train results to DataFrame: {e}")
        return None  # Handle the case where conversion fails

    return df_train_results

def _parallel_evaluate(model_name, X_train, y_train, groups_train):
    # define number of CPU cores
    num_models = 7 
    num_cores = -1#int(multiprocessing.cpu_count()/ (num_models*2))
    print("Number of cores used for parallel processing: " + str(num_cores))
    if model_name == 'RF':
        param_grid = {
            'n_estimators': [1, 2, 3, 5, 10, 30, 50, 100, 200, 300, 500]
            #'n_estimators': [100],  # Skip very small values like 1, 2, 3
            #'max_depth': [None, 10, 50],       # Control depth to avoid overfitting
            #'max_features': ['sqrt'],     # Experiment with feature subset sizes
            #'min_samples_split': [2, 10],           # Prevent overly complex splits
            #'min_samples_leaf': [1, 5], 
        }
        model = RandomForestClassifier(random_state=1234, n_jobs=num_cores) # use 24 cores for parallel processing
    elif model_name == 'SVM':
        param_grid = {'kernel': ['linear', 'rbf', 'poly', 'sigmoid'], 'gamma': [0.1, 0.01, 1e-3, 1e-4],
                      'C': [1, 10, 100, 1000]}
        model = SVC(probability=True, random_state=1234)
    elif model_name == 'NB':
        param_grid = {'var_smoothing': np.logspace(0, -9, num=100)}
        model = GaussianNB()
    elif model_name == 'MLP':
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd'],
            'learning_rate': ['constant', 'adaptive'],
        }
        model = MLPClassifier(random_state=1234, max_iter=500, tol=1e-4, learning_rate_init=0.001, early_stopping=True, validation_fraction=0.1)
    elif model_name == 'KNN':
        param_grid = {
            'n_neighbors': (1, 10, 1),
            'leaf_size': (20, 40, 1),
            'p': (1, 2),
            'weights': ('uniform', 'distance'),
            'metric': ('minkowski', 'chebyshev')
        }
        model = KNeighborsClassifier()  # XXX removed random_state=1234
    elif model_name == 'LR':
        param_grid = {"C": np.logspace(-3, 3, 7), "penalty": ["l2"]} # "l1" removed
        model = LogisticRegression(random_state=1234, max_iter=500)
    elif model_name == 'XGboost':
        param_grid = {
            'min_child_weight': [1, 5, 10],
            'gamma': [0.5, 1, 1.5, 2, 5],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'max_depth': [3, 4, 5]
        }
        model = XGBClassifier(random_state=1234)

    metrics = _evaluate_model(model_name, model, param_grid, X_train, y_train, groups_train)

    return metrics

    #return {
    #    'Model': model_name,
    #    'AUC': roc_mean,
    #    'AUC std': std_roc,
    #    'Accuracy': accuracy_mean,
    #    'Accuracy std': std_accuracy,
    #    'Recall': recall_mean,
    #    'Recall std': std_recall,
    #    'Precision': precision_mean,
    #    'Precision std': std_precision,
    #    'clf_best': clf
    #}

def _evaluate_model(model_name, model, param_grid, X, y, groups):
    """
    Evaluate the model using grouped cross-validation within GridSearchCV.

    Args:
        model_name: the name of the machine learning model
        model: The machine learning model to evaluate.
        param_grid: The hyperparameter grid for GridSearchCV.
        X: Feature matrix.
        y: Labels.
        groups: Group identifiers for Grouped CrossValidation.

    Returns:
        Median, IQR, min, and max for ROC, accuracy, recall, precision, F1 score, and best model.
    """
    # Initialize cross-validator
    cv = StratifiedGroupKFold(n_splits=5)

    # Scoring metrics
    scoring = {
        'AUC': make_scorer(roc_auc_score),
        'Accuracy': make_scorer(accuracy_score),
        'Recall': make_scorer(recall_score),
        'Precision': make_scorer(precision_score, average='weighted', zero_division=0),
        'F1': make_scorer(f1_score, average='weighted')
    }

    # Use GridSearchCV to find the best model
    clf = GridSearchCV(
        model,
        param_grid,
        scoring=scoring,
        refit='AUC',  # Select the model based on ROC AUC score
        cv=cv,
        return_train_score=False,
        verbose=3, # Verbose output for debugging
        n_jobs=-1,  # Use all available cores for parallel processing
        pre_dispatch='2*n_jobs'  # Manage memory usage
    )

    # Fit the model with the given cross-validation scheme
    clf.fit(X, y, groups=groups)

    # Retrieve the best estimator
    clf_best = clf.best_estimator_

    # Extract cross-validation results
    cv_results = clf.cv_results_

    # Initialize metrics dictionary
    metrics = {'Model': model_name, 'clf_best': clf_best}  # Add model and clf as initial keys


    for metric in ['AUC', 'Accuracy', 'Recall', 'Precision', 'F1']:
        # Extract scores
        scores = cv_results[f'mean_test_{metric}']

        mean_score = np.mean(scores)
        std_score = np.std(scores)

        # Calculate median
        median_score = np.median(scores)

        # Calculate IQR
        q75, q25 = np.percentile(scores, [75, 25])
        iqr_score = q75 - q25

        # Calculate min and max
        min_score = np.min(scores)
        max_score = np.max(scores)

        metrics[metric + '_Mean'] = mean_score
        metrics[metric + '_Std'] = std_score
        metrics[metric + '_Median'] = median_score
        metrics[metric + '_IQR'] = iqr_score
        metrics[metric + '_Min'] = min_score
        metrics[metric + '_Max'] = max_score

    return metrics

def plot_test_results(df_test_results):
    # Group by Model and calculate median and min/max
    summary = df_test_results.groupby('Model').agg(['mean', 'std']).reset_index()
    summary.columns = ['Model'] + [f"{metric}_{stat}" for metric, stat in summary.columns[1:]]

    # Prepare data for plotting
    models = summary['Model']
    metrics = ['AUC', 'Accuracy', 'Recall', 'Precision', 'F1']

    # Initialize a bar plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Define bar width and positions
    bar_width = 0.15
    x = np.arange(len(models))

    # Create a color array using the viridis colormap
    cmap = cm.get_cmap('viridis', len(metrics))

    # Plot each metric
    for i, metric in enumerate(metrics):
        mean_values = summary[f"{metric}_mean"]
        std_values = summary[f"{metric}_std"]

        # Define bar positions
        bar_positions = x + i * bar_width

        # Plot bars for the median values
        ax.bar(bar_positions, mean_values, width=bar_width,
               label=metric, color=cmap(i))

        # Add error bars for min/max
        ax.errorbar(bar_positions, mean_values,
                    yerr= std_values, #[median_values - min_values, max_values - median_values],
                    fmt='none', color='black', capsize=5)

    # Configure axes and labels
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Scores', fontsize=12)
    ax.set_title('Model Performance Metrics', fontsize=14)
    ax.set_xticks(x + bar_width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(models)
    ax.legend(title='Metrics')

    # Show plot
    plt.tight_layout()
    #plt.show()
    return plt.gcf()

def save_df_results_to_hd(target_path, df_train_results, df_test_results, X_train_list, y_train_list, X_test_list,
                          y_test_list):
    os.makedirs(target_path, exist_ok=True)

    # save lists as pickles format
    full_path = os.path.join(target_path, 'X_train_list.pkl')
    with open(full_path, 'wb') as f:
        pickle.dump(X_train_list, f)

    full_path = os.path.join(target_path, 'y_train_list.pkl')
    with open(full_path, 'wb') as f:
        pickle.dump(y_train_list, f)

    full_path = os.path.join(target_path, 'X_test_list.pkl')
    with open(full_path, 'wb') as f:
        pickle.dump(X_test_list, f)

    full_path = os.path.join(target_path, 'y_test_list.pkl')
    with open(full_path, 'wb') as f:
        pickle.dump(y_test_list, f)

    # save dataframe
    full_path = os.path.join(target_path, 'df_train_results.pkl')
    # df_results.to_csv(full_path, index=False)
    print("Saving: " + full_path)
    df_train_results.to_pickle(full_path)

    full_path = os.path.join(target_path, 'df_test_results.pkl')
    df_test_results.to_pickle(full_path)

def load_df_results_from_hd(ml_result_path):

    full_path = os.path.join(ml_result_path, 'X_train_list.pkl')
    with open(full_path, 'rb') as f:
        X_train_list = pickle.load(f)

    full_path = os.path.join(ml_result_path, 'y_train_list.pkl')
    with open(full_path, 'rb') as f:
        y_train_list = pickle.load(f)

    full_path = os.path.join(ml_result_path, 'X_test_list.pkl')
    with open(full_path, 'rb') as f:
        X_test_list = pickle.load(f)

    full_path = os.path.join(ml_result_path, 'y_test_list.pkl')
    with open(full_path, 'rb') as f:
        y_test_list = pickle.load(f)

    full_path_pkl = os.path.join(ml_result_path, 'df_train_results.pkl')
    df_train_results = pd.read_pickle(full_path_pkl)

    full_path_pkl = os.path.join(ml_result_path, 'df_test_results.pkl')
    df_test_results = pd.read_pickle(full_path_pkl)

    return df_train_results, df_test_results, X_train_list, y_train_list, X_test_list, y_test_list

###############################


def plot_and_analyze_df_results(df_results, X_test, y_test):
    fig_train = plot_results_train(df_results)
    fig_test = plot_results_test(df_results, X_test, y_test)
    return fig_train, fig_test


def print_result_in_terminal(df_results, X_test, y_test):
    # Print the results
    print('Train set performance')
    print(df_results)

    # Test set performance for the best model
    for model, best_model in zip(df_results['Model'], df_results['clf_best']):
        # best_model = df_results.loc[df_results['AUC'].idxmax()]['clf_best']
        # y_test1 = y_test.to_numpy()
        #  y_test1 = y_test1.reshape(len(y_test1),)
        n_classes = len(np.unique(y_test))
        y_test1 = label_binarize(y_test, classes=np.arange(n_classes))

        print(model)
        y_pred_test = best_model.predict(X_test)
        y_pred1 = label_binarize(y_pred_test, classes=np.arange(n_classes))
        print('Test set performance')
        print('AUC test:', roc_auc_score(y_test, y_pred_test))
        print('Accuracy test:', accuracy_score(y_test, y_pred_test))
        print('F1 score test:', f1_score(y_test, y_pred_test, average="macro", pos_label=0))
        print('Recall test:', recall_score(y_test, y_pred_test, average="macro", pos_label=0))
        print('Precision test:', precision_score(y_test, y_pred_test, average="macro", pos_label=0))

        print(classification_report(y_test, y_pred_test))


def plot_results_train(df_results):
    # Your existing code to load data and set variables...

    sns.set_palette("viridis")

    models = df_results['Model']
    metrics = ['AUC', 'Accuracy', 'Recall', 'Precision']
    error_metrics = ['AUC std', 'Accuracy std', 'Recall std', 'Precision std']

    fig, ax = plt.subplots(figsize=(12, 6))  # Adjust the figure size as needed
    ax.set_xlabel("ML algorithms")
    ax.set_ylabel("Values")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    bar_width = 0.15
    positions = np.arange(len(models))

    for i, metric in enumerate(metrics):
        values = df_results[metric]
        errors = df_results[error_metrics[i]]
        # Adjusting x positions to center the bars for each group
        ax.bar(positions + i * bar_width - 0.3, values, bar_width, label=metric, yerr=errors, alpha=0.8, bottom=0)

    ax.set_xticks(np.arange(len(models)) + ((len(metrics) - 1) / 2 + 0.5) * bar_width)
    ax.set_xticklabels(models)
    ax.legend()

    # Increase font size for axis labels
    ax.set_xlabel("ML algorithms", fontsize=16)
    ax.set_ylabel("Values", fontsize=16)

    # Increase font size for tick labels on both axes
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    # Increase font size for legend
    ax.legend(fontsize=12)

    # Adjust x-axis limits to include the entire graph
    ax.set_xlim(-0.5, len(models) + 0.5)
    # Align the minor tick label
    for label in ax.get_xticklabels(minor=True):
        label.set_horizontalalignment('center')
    # Set y-axis limits
    ax.set_ylim(0, 1.1)

    # Adjust legend position
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=2)

    plt.tight_layout()
    # plt.savefig('plot-train-sync.pdf', bbox_inches='tight', dpi=3000)
    # plt.show()

    return fig


def calculate_test_metrics(df_results, X_test, y_test):
    list_auc_test = []
    list_accuracy_test = []
    list_precision_test = []
    list_recall_test = []
    for model, best_model in zip(df_results['Model'], df_results['clf_best']):
        # best_model = df_results.loc[df_results['AUC'].idxmax()]['clf_best']
        # y_test1 = y_test.to_numpy()
        #  y_test1 = y_test1.reshape(len(y_test1),)
        n_classes = len(np.unique(y_test))
        # MC: y_test1 = label_binarize(y_test, classes=np.arange(n_classes))

        print(model)
        y_pred_test = best_model.predict(X_test)
        # MC: y_pred1 = label_binarize(y_pred_test, classes=np.arange(n_classes))
        print('Test set performance')
        print('AUC test:', roc_auc_score(y_test, y_pred_test))
        list_auc_test.append(roc_auc_score(y_test, y_pred_test))
        print('Accuracy test:', accuracy_score(y_test, y_pred_test))
        list_accuracy_test.append(accuracy_score(y_test, y_pred_test))

        print('F1 score test:', f1_score(y_test, y_pred_test, average="macro", pos_label=0))
        print('Recall test:', recall_score(y_test, y_pred_test, average="macro", pos_label=0))
        list_recall_test.append(recall_score(y_test, y_pred_test, average="macro", pos_label=0))

        print('Precision test:', precision_score(y_test, y_pred_test, average="macro", pos_label=0))
        list_precision_test.append(precision_score(y_test, y_pred_test, average="macro", pos_label=0))
        print(classification_report(y_test, y_pred_test))

    return list_auc_test, list_accuracy_test, list_recall_test, list_precision_test


def plot_results_test(df_results, X_test, y_test):
    list_auc_test, list_accuracy_test, list_recall_test, list_precision_test = calculate_test_metrics(df_results,
                                                                                                      X_test, y_test)

    fig = call_plot_test_metrics(df_results, list_auc_test, list_accuracy_test, list_recall_test, list_precision_test)
    return fig


def call_plot_test_metrics(df_results, list_auc_test, list_accuracy_test, list_recall_test, list_precision_test):
    # Plotting test metrics
    test_metrics = ['AUC', 'Accuracy', 'Recall', 'Precision']
    test_values = [list_auc_test, list_accuracy_test, list_recall_test, list_precision_test]

    fig, ax = plot_test_metrics(df_results, test_values, test_metrics)
    # plt.savefig('plot-test-sync.pdf', bbox_inches='tight', dpi=3000)
    # plt.show()
    return fig


def plot_test_metrics(df_results, test_values, test_metrics):
    models = df_results['Model']
    metrics = ['AUC', 'Accuracy', 'Recall', 'Precision']
    error_metrics = ['AUC std', 'Accuracy std', 'Recall std', 'Precision std']

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel("ML algorithms")
    ax.set_ylabel("Values")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    bar_width = 0.15
    positions = np.arange(len(models))

    for i, metric in enumerate(metrics):
        values = df_results[metric]
        errors = df_results[error_metrics[i]]
        ax.bar(positions + i * bar_width, values, bar_width, label=metric, alpha=0.8, bottom=0)
        #    ax.bar(positions + (len(metrics) + i) * bar_width, bar_width, label=metric, yerr=errors, alpha=0.8, bottom=0)
        # ax.set_xticks(positions + bar_width * (len(metrics) - 1) / 2)
        ax.set_xticks(np.arange(len(models)) + ((len(test_metrics) - 1) / 2 + 0.5) * bar_width)

    # Align the minor tick label
    for label in ax.get_xticklabels(minor=True):
        label.set_horizontalalignment('center')
    ax.set_xticklabels(models)
    ax.legend()
    # ax.set_xticks(positions + bar_width * ((len(metrics) - 1) / 2 + len(test_metrics)))
    # ax.set_xticklabels(models)
    # After setting x-axis ticks and labels

    # ax.legend()

    # Rest of the formatting - font sizes, limits, etc.
    ax.set_xlabel("ML algorithms", fontsize=16)
    ax.set_ylabel("Values", fontsize=16)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(fontsize=12)
    # for label in ax.get_xticklabels(minor=True):
    #    label.set_horizontalalignment('center')
    ax.set_xticks(np.arange(len(models)) + ((len(test_metrics) - 1) / 2) * bar_width)
    ax.set_xticklabels(models)

    # ax.set_xlim(-bar_width, len(models) - 1 + (len(metrics) + len(test_metrics) - 1) * bar_width)
    ax.set_ylim(0, 1.1)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=2)
    plt.tight_layout()

    return fig, ax
