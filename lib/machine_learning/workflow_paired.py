import glob
from multiprocessing import Pool
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
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from xgboost import XGBClassifier

from lib.connectivity import matrix_handling


def call(models, df_features_class0, df_features_class1):
    warnings.filterwarnings("ignore")

    # Step 1: scale the data (pairwise)
    df_features_class0_scaled, df_features_class1_scaled = _pairwise_scale(df_features_class0, df_features_class1)

    # Step 2: Get labels (0 for class0, 1 for class1)
    y_class0 = [0] * len(df_features_class0_scaled)
    y_class1 = [1] * len(df_features_class1_scaled)
    y = y_class0 + y_class1
    y = pd.DataFrame(y)

    # Combine scaled data and create group labels (e.g., chip IDs)
    X = pd.concat([df_features_class0_scaled, df_features_class1_scaled], axis=0).reset_index(drop=True)
    # Remove chip_id column from the feature data!!!
    X.drop('chip_id', axis=1, inplace=True)  # remove chip_id column
    # Remove columns where all values are zero
    X = X.loc[:, (X != 0).any(axis=0)]

    # create group lables
    groups_class0 = df_features_class0['chip_id'].values  # Assuming 'chip_id' is the column
    groups_class1 = df_features_class1['chip_id'].values
    groups = np.concatenate([groups_class0, groups_class1])

    # Step 3: Outer Cross-Validation using Leave-One-Group-Out
    logo = LeaveOneGroupOut()
    train_result_list = []  # To store the training metrics (inner CV results)
    test_result_list = []   # To store the test set performance (outer test results)
    X_train_list = [] # to store X_train for all splits, needed for SHAP
    y_train_list = []  # to store y_train for all splits, needed for SHAP
    X_test_list = []
    y_test_list = []

    cnt = 0
    for train_idx, test_idx in logo.split(X, y, groups):

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

def _pairwise_scale(df_features_class0, df_features_class1):
    """
    Scale features of class 0 (pre-drug) and class 1 (post-drug)
    for each chip, using the mean and std of the pre-drug values (class 0).

    Args:
        df_features_class0: Pandas DataFrame for pre-drug features (class 0)
        df_features_class1: Pandas DataFrame for post-drug features (class 1)

    Returns:
        df_scaled_class0: Scaled features for class 0.
        df_scaled_class1: Scaled features for class 1.
    """

    df_scaled_class0 = pd.DataFrame()
    df_scaled_class1 = pd.DataFrame()
    # df test just used for debug
    #df_test = pd.DataFrame()


    # Iterate through each chip and scale
    for chip_id in df_features_class0['chip_id'].unique():
        # Select pre-drug (class 0) and post-drug (class 1) data for this chip
        chip_data_class0 = df_features_class0[df_features_class0['chip_id'] == chip_id]
        chip_data_class1 = df_features_class1[df_features_class1['chip_id'] == chip_id]

        # Remove the 'chip_id' column temporarily
        chip_data_class0_no_id = chip_data_class0.drop(columns=['chip_id'])
        chip_data_class1_no_id = chip_data_class1.drop(columns=['chip_id'])

        # Create and fit the scaler on class 0 (pre-drug) data
        scaler = preprocessing.StandardScaler()
        scaler.fit(chip_data_class0_no_id)

        # Scale both class 0 and class 1 using the class 0 mean and std
        chip_data_class0_scaled = pd.DataFrame(scaler.transform(chip_data_class0_no_id),
                                               columns=chip_data_class0_no_id.columns)
        chip_data_class1_scaled = pd.DataFrame(scaler.transform(chip_data_class1_no_id),
                                               columns=chip_data_class1_no_id.columns)

        # Add back chip_id column for concatenation
        chip_data_class0_scaled['chip_id'] = chip_id
        chip_data_class1_scaled['chip_id'] = chip_id

        # Append scaled data for this chip
        df_scaled_class0 = pd.concat([df_scaled_class0, chip_data_class0_scaled])
        df_scaled_class1 = pd.concat([df_scaled_class1, chip_data_class1_scaled])
        # df_test just used for debug
        #df_test = pd.concat([df_test, chip_data_class0_scaled, chip_data_class1_scaled])

    # Reset index for the full dataframe
    df_scaled_class0 = df_scaled_class0.reset_index(drop=True)
    df_scaled_class1 = df_scaled_class1.reset_index(drop=True)

    df_scaled_class0.replace(np.nan, 0, inplace=True)
    df_scaled_class1.replace(np.nan, 0, inplace=True)

    return df_scaled_class0, df_scaled_class1

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
    if model_name == 'RF':
        param_grid = {
            'n_estimators': [1, 2, 3, 5, 10, 30, 50, 100, 200, 300, 500]
        }
        model = RandomForestClassifier(random_state=1234)
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
    Evaluate the model using Leave-One-Group-Out cross-validation within GridSearchCV.

    Args:
        model_name: the name of the machine learning model
        model: The machine learning model to evaluate.
        param_grid: The hyperparameter grid for GridSearchCV.
        X: Feature matrix.
        y: Labels.
        groups: Group identifiers for Leave-One-Group-Out.

    Returns:
        Median, IQR, min, and max for ROC, accuracy, recall, precision, F1 score, and best model.
    """
    # Initialize Leave-One-Group-Out cross-validator
    logo = LeaveOneGroupOut()

    # Scoring metrics
    scoring = {
        'AUC': make_scorer(roc_auc_score),
        'Accuracy': make_scorer(accuracy_score),
        'Recall': make_scorer(recall_score),
        'Precision': make_scorer(precision_score, average='weighted', zero_division=0),
        'F1': make_scorer(f1_score, average='weighted')
    }

    # Use GridSearchCV with LeaveOneGroupOut to find the best model
    clf = GridSearchCV(
        model,
        param_grid,
        scoring=scoring,
        refit='AUC',  # Select the model based on ROC AUC score
        cv=logo,
        return_train_score=False
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

def calculate_shap(df_train_result, X_train, y_train):
    pass

###############################

def call_old(models, df_features_bic00, df_features_class1):
    warnings.filterwarnings("ignore")

    X_train, X_test, y_train, y_test, groups_train, groups_test = scale_and_split(df_features_bic00,
                                                                                        df_features_class1)
    df_results, X_test, y_test = calling_the_models(models, X_train, X_test, y_train, y_test, groups_train)

    return df_results, X_test, y_test


def load_csv_to_np(full_path):
    matrix = np.array(pd.read_csv(full_path, header=None))
    # matrix = pd.read_csv(full_path, index=None)
    return matrix


def load_all_matrices(experiment_path, group, index_col=0):
    # group: "class0" or "class1"

    matrices_list = []

    # for all chips in current folder
    chip_list = os.listdir(experiment_path)
    for chip in chip_list:

        path_group = os.path.join(experiment_path, chip, group)
        file_list = sorted(glob.glob(path_group + '/*.csv'))

        # for all files (=windows) of the current chip
        for file in file_list:
            matrices_list.append(matrix_handling.load_correlation_matrix(file, index_col))

    return matrices_list


# TODO: is it correct to replace nan by zeros?
# This is called before matrix gets fed into the ML model
def flatten_matrices(matrix_list):
    replaced_list = []
    flatten_list = []

    for i in matrix_list:
        replaced_list.append(np.nan_to_num(i))

    flatten_list = replaced_list
    # print(len(replaced_list ))
    # print(len(flatten_list))
    return flatten_list


def _flatten_matrix(x):
    flatten = matrix.flatten(np.array(x))
    return flatten


# TODO: is it correct to replace nan by zeros?
def flatten_df_matrices(df_matrix_list):
    replaced_list = []
    flatten_list = []

    for df in df_matrix_list:
        replaced_list.append(df.fillna(0))

    for df in replaced_list:
        flatten_list.append(matrix_handling.flatten_df_matrix(df))

    return flatten_list


def _get_labels(df_features_class0, df_features_class1):
    no_label = [0] * len(df_features_class0)
    yes_label = [1] * len(df_features_class1)

    labels = no_label + yes_label

    y = np.nan_to_num(labels)


    return y

def scale_and_split(df_features_class0, df_features_class1, flag_scaling=True):
    # Get labels for both groups
    y = _get_labels(df_features_class0, df_features_class1)

    # Combine both feature sets into one DataFrame
    X = pd.concat([df_features_class0, df_features_class1], axis=0).reset_index(drop=True)

    # Create group labels based on which chip the data comes from
    # Assuming you have chip IDs in a column, e.g., "chip_id"
    groups_class0 = df_features_class0['chip_id'].values
    groups_class1 = df_features_class1['chip_id'].values

    # Combine the group labels from both conditions (must align with X and y)
    groups = np.concatenate([groups_class0, groups_class1])

    # GroupShuffleSplit to split data, ensuring no leakage of chips between train and test
    gss = GroupShuffleSplit(test_size=0.25, n_splits=1, random_state=1234)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    # Get train and test sets
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train, groups_test = groups[train_idx], groups[test_idx]

    print("Is there an inbalance between classes?")
    print(np.bincount(y_train))
    print(np.bincount(y_test))

    # do the scaling seperate for training and test to avoid data leackage
    if flag_scaling:
        scaler = preprocessing.StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        X_train = X_train.to_numpy()  # to numpy array like the fit_transform function does
        X_test = X_test.to_numpy()

    # Convert NaNs to numbers if present
    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)

    return X_train, X_test, y_train, y_test, groups_train, groups_test



def pairwise_scaleOLD(df_features_class0, df_features_class1):
    """
    scale features of class 0 (pre-drug) and class 1 (post-drug)
    for each chip, using the mean and std of the pre-drug values (class 0).

    Args:
        df_features_class0: Pandas DataFrame for pre-drug features (class 0)
        df_features_class1: Pandas DataFrame for post-drug features (class 1)

    Returns:
        df_scaled: A DataFrame containing the scaled features for both classes.
    """

    # replace NaN with zeros
    #df_features_class0.fillna(0, inplace=True)
    #df_features_class1.fillna(0, inplace=True)

    df_scaled_class0 = pd.DataFrame()
    df_scaled_class1 = pd.DataFrame()

    # Iterate through each chip and scale
    for chip_id in df_features_class0['chip_id'].unique():
        # Select pre-drug (class 0) and post-drug (class 1) data for this chip
        chip_data_class0 = df_features_class0[df_features_class0['chip_id'] == chip_id]
        chip_data_class1 = df_features_class1[df_features_class1['chip_id'] == chip_id]

        # Compute mean and std for class 0 (pre-drug)
        mean_class0 = chip_data_class0.mean()
        std_class0 = chip_data_class0.std()

        # scale both class 0 and class 1 using the class 0 mean and std
        chip_data_class0_scaled = (chip_data_class0 - mean_class0) / std_class0
        chip_data_class1_scaled = (chip_data_class1 - mean_class0) / std_class0

        # Add back chip_id column for concatenation
        chip_data_class0_scaled['chip_id'] = chip_id
        chip_data_class1_scaled['chip_id'] = chip_id

        # Append scaled data for this chip
        #df_scaled = pd.concat([df_scaled, chip_data_class0_scaled, chip_data_class1_scaled])
        df_scaled_class0 = pd.concat([df_scaled_class0, chip_data_class0_scaled])
        df_scaled_class1 = pd.concat([df_scaled_class1, chip_data_class1_scaled])

    # Reset index for the full dataframe
    df_scaled_class0 = df_scaled_class0.reset_index(drop=True)
    df_scaled_class1 = df_scaled_class1.reset_index(drop=True)

    return df_scaled_class0, df_scaled_class1



def _evaluate_model_old(model, param_grid, X, y, groups):
    # note, here only the data of the train data set is split into train/test during the
    # cross validation

    # instead of kfold crossvalidatioin, logo is used:
    logo = LeaveOneGroupOut()  # Initialize Leave-One-Group-Out
    recall = []
    precision = []
    f = []
    accuracy = []
    roc = []

    for train_idx, test_idx in logo.split(X, y, groups):
        xtr, xvl = X.iloc[train_idx], X.iloc[test_idx]
        ytr, yvl = y.iloc[train_idx], y.iloc[test_idx]

        ytr = ytr.values.ravel()
        clf = GridSearchCV(model, param_grid, scoring='roc_auc')
        clf.fit(xtr, ytr)
        clf_best = clf.best_estimator_

        y_pred = clf_best.predict(xvl)

        yvl = yvl.values.ravel()
        n_classes = len(np.unique(yvl))
        yvl1 = label_binarize(yvl, classes=np.arange(n_classes))
        y_pred1 = label_binarize(y_pred, classes=np.arange(n_classes))

        roc.append(roc_auc_score(yvl, y_pred))
        accuracy.append(accuracy_score(yvl, y_pred))
        recall.append(recall_score(yvl, y_pred))
        precision.append(precision_score(yvl, y_pred, average='weighted', zero_division=0))

        f.append(f1_score(yvl, y_pred, average='weighted'))

    return (np.mean(roc), np.std(roc), np.mean(accuracy), np.std(accuracy),
            np.mean(recall), np.std(recall), np.mean(precision), np.std(precision), clf_best)


def _evaluate_model_kfold(model, param_grid, X, y, groups):
    gkf = GroupKFold(n_splits=10)  # Use GroupKFold for cross-validation with groups
    recall = []
    precision = []
    f = []
    accuracy = []
    roc = []

    # Iterate over each split from GroupKFold
    for train_index, test_index in gkf.split(X, y, groups=groups):
        xtr, xvl = X.loc[train_index], X.loc[test_index]
        ytr, yvl = y.loc[train_index], y.loc[test_index]

        ytr = ytr.to_numpy().reshape(len(ytr), )
        clf = GridSearchCV(model, param_grid, scoring='roc_auc')
        clf.fit(xtr, ytr)
        clf_best = clf.best_estimator_

        y_pred = clf_best.predict(xvl)

        yvl = yvl.to_numpy().reshape(len(yvl), )
        n_classes = len(np.unique(yvl))
        yvl1 = label_binarize(yvl, classes=np.arange(n_classes))
        y_pred1 = label_binarize(y_pred, classes=np.arange(n_classes))

        roc.append(roc_auc_score(yvl, y_pred))
        accuracy.append(accuracy_score(yvl, y_pred))
        recall.append(recall_score(yvl, y_pred))
        precision.append(precision_score(yvl, y_pred, average='weighted'))

        f.append(f1_score(yvl, y_pred, average='weighted'))

    return (np.mean(roc), np.std(roc), np.mean(accuracy), np.std(accuracy),
            np.mean(recall), np.std(recall), np.mean(precision), np.std(precision), clf_best)


def _evaluate_model_old_old(model, param_grid, X, y):
    skf = StratifiedKFold(n_splits=10)
    recall = []
    precision = []
    f = []
    accuracy = []
    roc = []

    for train_index, test_index in skf.split(X, y):
        xtr, xvl = X.loc[train_index], X.loc[test_index]
        ytr, yvl = y.loc[train_index], y.loc[test_index]

        ytr = ytr.to_numpy().reshape(len(ytr), )
        clf = GridSearchCV(model, param_grid, scoring='roc_auc')
        clf.fit(xtr, ytr)
        clf_best = clf.best_estimator_

        y_pred = clf.best_estimator_.predict(xvl)

        yvl = yvl.to_numpy().reshape(len(yvl), )
        n_classes = len(np.unique(yvl))
        yvl1 = label_binarize(yvl, classes=np.arange(n_classes))
        y_pred1 = label_binarize(y_pred, classes=np.arange(n_classes))

        roc.append(roc_auc_score(yvl, y_pred))
        accuracy.append(accuracy_score(yvl, y_pred))
        recall.append(recall_score(yvl, y_pred))
        precision.append(precision_score(yvl, y_pred, average='weighted'))

        f.append(sklearn.metrics.f1_score(yvl, y_pred, average='weighted'))

    return (np.mean(roc), np.std(roc), np.mean(accuracy), np.std(accuracy),
            np.mean(recall), np.std(recall), np.mean(precision), np.std(precision), clf_best)



def save_df_results_to_HD_OLD(target_path, df_results, X_test, y_test):
    os.makedirs(target_path, exist_ok=True)

    # save numpy arrays
    full_path = os.path.join(target_path, 'X_test_y_test.npz')

    np.savez(full_path, arr1=X_test, arr2=y_test)
    # HOW TO LOAD:
    # data = np.load('my_arrays.npz')
    # arr1_loaded = data['arr1']
    # arr2_loaded = data['arr2']

    # save dataframe
    print("Saving: " + full_path)
    full_path = os.path.join(target_path, 'df_results.pkl')
    # df_results.to_csv(full_path, index=False)
    df_results.to_pickle(full_path)


def load_df_results_from_hd_OLD(ml_result_path):
    full_path_np = os.path.join(ml_result_path, 'X_test_y_test.npz')
    data = np.load(full_path_np)
    X_test = data['arr1']
    y_test = data['arr2']

    full_path_pkl = os.path.join(ml_result_path, 'df_results.pkl')
    df_results = pd.read_pickle(full_path_pkl)

    return df_results, X_test, y_test


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
