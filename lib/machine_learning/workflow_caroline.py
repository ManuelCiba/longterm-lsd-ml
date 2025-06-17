import glob
import pandas as pd
import math
import glob
import numpy as np
from pandas import DataFrame
from numpy import matrix
from sklearn.model_selection import StratifiedKFold
from sklearn import model_selection
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import seaborn as sns
from yellowbrick.model_selection import LearningCurve
import matplotlib.pyplot as plt
from scipy import interpolate as interp #interp
from itertools import cycle
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
import joblib
from sklearn.metrics import roc_curve, auc, roc_auc_score,accuracy_score,f1_score,recall_score,precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
import joblib
from sklearn.metrics import roc_curve, auc, roc_auc_score,accuracy_score,f1_score,recall_score,precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn import model_selection
from sklearn import svm
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,  precision_score, recall_score, classification_report, confusion_matrix
import sklearn.metrics
import multiprocessing
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import warnings
import os
from lib.connectivity import matrix_handling

def load_csv_to_np(full_path):
    matrix = np.array(pd.read_csv(full_path, header=None))
    #matrix = pd.read_csv(full_path, index=None)
    return matrix


def load_all_matrices(experiment_path, group, index_col=0):
    # group: "bic00" or "bic10"

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


def _flatten_matrix(x):
    flatten = matrix.flatten(np.array(x))
    return flatten

# TODO: is it correct to replace nan by zeros?
# This is called before matrix gets fed into the ML model
def flatten_matrices(matrix_list):
    replaced_list = []
    flatten_list = []

    for i in matrix_list:
        replaced_list.append(np.nan_to_num(i))

    for i in replaced_list:
        flatten_list.append(_flatten_matrix(i))

    # print(len(replaced_list ))
    # print(len(flatten_list))
    return flatten_list

# TODO: is it correct to replace nan by zeros?
def flatten_df_matrices(df_matrix_list):
    replaced_list = []
    flatten_list = []

    for df in df_matrix_list:
        replaced_list.append(df.fillna(0))

    for df in replaced_list:
        flatten_list.append(matrix_handling.flatten_df_matrix(df))

    return flatten_list


def _get_labels(flatten_list_bic00, flatten_list_bic10):

    no_label = [0] * len(flatten_list_bic00)
    yes_label = [1] * len(flatten_list_bic10)

    labels = yes_label + no_label
    #
    # print(len(labels))
    y = np.nan_to_num(labels)
    return y


def standardize_and_split(flatten_list_bic00, flatten_list_bic10):

    y = _get_labels(flatten_list_bic00, flatten_list_bic10)

    myScaler = preprocessing.StandardScaler()
    X = np.concatenate([flatten_list_bic10, flatten_list_bic00], axis=0)
    X = myScaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, shuffle=True,
                                                        random_state=1234)
    X_train = np.nan_to_num(X_train)

    X_test = np.nan_to_num(X_test)
    # print(np.shape(X_train))
    # print(np.shape(X_test))
    return X_train, X_test, y_train, y_test


def _evaluate_model(model, param_grid, X, y):
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


def parallel_evaluate(model_name, X_train, y_train):
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
        model = MLPClassifier(random_state=1234)
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
        param_grid = {"C": np.logspace(-3, 3, 7), "penalty": ["l1", "l2"]}
        model = LogisticRegression(random_state=1234)
    elif model_name == 'XGboost':
        param_grid = {
            'min_child_weight': [1, 5, 10],
            'gamma': [0.5, 1, 1.5, 2, 5],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'max_depth': [3, 4, 5]
        }
        model = XGBClassifier(random_state=1234)

    roc_mean, std_roc, accuracy_mean, std_accuracy, recall_mean, std_recall, precision_mean, std_precision, clf = _evaluate_model(
        model, param_grid, X_train, y_train)

    return {
        'Model': model_name,
        'AUC': roc_mean,
        'AUC std': std_roc,
        'Accuracy': accuracy_mean,
        'Accuracy std': std_accuracy,
        'Recall': recall_mean,
        'Recall std': std_recall,
        'Precision': precision_mean,
        'Precision std': std_precision,
        'clf_best': clf
    }


def calling_the_models(models, X_train, X_test, y_train, y_test):
    X_train = pd.DataFrame(X_train)
    y_train = pd.DataFrame(y_train)
    pool = multiprocessing.Pool(processes=len(models))
    results = pool.starmap(parallel_evaluate, [(model, X_train, y_train) for model in models])
    pool.close()
    pool.join()

    # Create a DataFrame to store the results
    df_results = pd.DataFrame(results)

    return df_results, X_test, y_test


def caroline_workflow(models, matrix_list_bic00, matrix_list_bic10):
    warnings.filterwarnings("ignore")

    flatten_list_bic00 = flatten_matrices(matrix_list_bic00)
    flatten_list_bic10 = flatten_matrices(matrix_list_bic10)
    X_train, X_test, y_train, y_test = standardize_and_split(flatten_list_bic00, flatten_list_bic10)

    df_results, X_test, y_test = calling_the_models(models, X_train, X_test, y_train, y_test)

    return df_results, X_test, y_test


def save_df_results_to_HD(target_path, df_results, X_test, y_test):

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
    #df_results.to_csv(full_path, index=False)
    df_results.to_pickle(full_path)


def load_df_results_from_hd(ml_result_path):
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
    #plt.savefig('plot-train-sync.pdf', bbox_inches='tight', dpi=3000)
    #plt.show()

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

    list_auc_test, list_accuracy_test, list_recall_test, list_precision_test = calculate_test_metrics(df_results, X_test, y_test)

    fig = call_plot_test_metrics(df_results, list_auc_test, list_accuracy_test, list_recall_test, list_precision_test)
    return fig


def call_plot_test_metrics(df_results, list_auc_test, list_accuracy_test, list_recall_test, list_precision_test):
    # Plotting test metrics
    test_metrics = ['AUC', 'Accuracy', 'Recall', 'Precision']
    test_values = [list_auc_test, list_accuracy_test, list_recall_test, list_precision_test]

    fig, ax = plot_test_metrics(df_results, test_values, test_metrics)
    #plt.savefig('plot-test-sync.pdf', bbox_inches='tight', dpi=3000)
    #plt.show()
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

# TODO, what is the purpose of this function? -> can be deleted?
def plot_results_train_test(df_results):
    sns.set_palette("viridis")
    models = df_results['Model']
    metrics = ['AUC', 'Accuracy', 'Recall', 'Precision']
    error_metrics = ['AUC std', 'Accuracy std', 'Recall std', 'Precision std']

    test_metrics = metrics  # TODO: Is this correct????

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel("ML algorithms")
    ax.set_ylabel("Values")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    bar_width = 0.15
    positions = np.arange(len(models))

    for i, metric in enumerate(metrics):
        values = df_results[metric]
        errors = df_results[error_metrics[i]]
        ax.bar(positions + i * bar_width, values, bar_width, label=metric, yerr=errors, alpha=0.8, bottom=0)
    # ax.bar(positions + (len(metrics) + i) * bar_width, bar_width, label=metric, yerr=errors, alpha=0.8, bottom=0)
    # ax.set_xticks(positions + bar_width * (len(metrics) - 1) / 2)
    ax.set_xticks(np.arange(len(models)) + ((len(test_metrics) - 1) / 2 + 0.5) * bar_width)

    # Align the minor tick label
    for label in ax.get_xticklabels(minor=True):
        label.set_horizontalalignment('center')
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
    # Adjust x-axis limits to align bars properly
    ax.set_xlim(-bar_width, len(models) - 1 + (len(metrics) - 1) * 1.5 * bar_width)
    # ax.set_xlim(-bar_width, (len(metrics) - 1) * bar_width)
    ax.set_ylim(0, 1.1)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=2)
    plt.tight_layout()
    plt.savefig('plot-train-sync.pdf', bbox_inches='tight', dpi=3000)
    plt.show()