import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import settings
import glob
import itertools  # generate all parameter combinations for parameter study


def get_all_chip_names():
    cwd = os.getcwd()
    path_data = cwd + '/data/MC-Japan/'

    path_chips = [x[0] for x in os.walk(path_data)]
    # remove first entry as it is the current folder
    path_chips = path_chips[1:]
    # sort by name
    path_chips = sorted(path_chips)

    chip_names = []
    for path_chip in path_chips:
        files = glob.glob(path_chip + '/*.mat')
        # Important! Sort file names, as python make wrong order
        files = sorted(files)

        # Split the path into its components
        parts = os.path.normpath(path_chip).split(os.path.sep)
        chip_name = parts[-1]

        chip_names.append(chip_name)

    return chip_names

def Xgenerate_all_paths(target_data_folder):
    # define parameter
    bin_sizes = settings.BIN_SIZES
    window_sizes = settings.WINDOW_SIZES
    window_overlaps = settings.WINDOW_OVERLAPS
    methods = settings.CONNECTIVITY_METHODS

    # Generate all combinations of parameters
    parameter_combinations = list(
        itertools.product(methods, bin_sizes, window_sizes, window_overlaps))

    # Base folder for storing results
    base_folder = os.path.join(settings.PATH_RESULTS_FOLDER, target_data_folder)

    folder_all_chips_list = []
    for combination in parameter_combinations:
        method, bin_size, window_size, window_overlap = combination

        # Create a folder structure based on the parameter combination
        result_folder = os.path.join(base_folder, f"method_{method}", f"bin_{bin_size}",
                                     f"window_{window_size}", f"overlap_{window_overlap}")

        folder_all_chips_list.append(result_folder)

    return folder_all_chips_list

def Xgenerate_all_paths_with_chip_name(target_data_folder):
    # define parameter
    bin_sizes = settings.BIN_SIZES
    window_sizes = settings.WINDOW_SIZES
    window_overlaps = settings.WINDOW_OVERLAPS
    methods = settings.CONNECTIVITY_METHODS
    groups = ["bic00", "bic10"]

    # get all chip names
    chip_names = get_all_chip_names()

    # Generate all combinations of parameters
    parameter_combinations = list(
        itertools.product(methods, bin_sizes, window_sizes, window_overlaps, chip_names, groups))

    # Base folder for storing results
    base_folder = os.path.join(settings.PATH_RESULTS_FOLDER, target_data_folder)

    folder_all_chips_list = []
    for combination in parameter_combinations:
        method, bin_size, window_size, window_overlap, chip_name, group = combination

        # Create a folder structure based on the parameter combination
        result_folder = os.path.join(base_folder, f"method_{method}", f"bin_{bin_size}",
                                     f"window_{window_size}", f"overlap_{window_overlap}", chip_name, group)

        folder_all_chips_list.append(result_folder)

    return folder_all_chips_list

def XXXgenerate_paths(target_data_folder, methods, bin_sizes, window_sizes, window_overlaps, chip_names, groups):

    # get all parameter combinations
    parameter_combinations = list(
        itertools.product(methods, bin_sizes, window_sizes, window_overlaps, chip_names, groups))

    # define Base folder for storing results
    base_folder = os.path.join(settings.PATH_RESULTS_FOLDER, target_data_folder)

    folder_all_chips_list = []
    for combination in parameter_combinations:
        method, bin_size, window_size, window_overlap, chip_name, group = combination
        # Create a folder structure based on the parameter combination
        result_folder = os.path.join(base_folder, f"method_{method}", f"bin_{bin_size}",
                                     f"window_{window_size}", f"overlap_{window_overlap}", chip_name, group)

        folder_all_chips_list.append(result_folder)

    return folder_all_chips_list

def generate_paths(target_data_folder, bin_sizes, window_sizes, window_overlaps, chip_names, groups):

    # Generate all combinations of parameters
    if chip_names != [] and groups != []:
        parameter_combinations = list(
            itertools.product(bin_sizes, window_sizes, window_overlaps, chip_names, groups))
    elif chip_names != [] and groups == []:
        parameter_combinations = list(
            itertools.product(bin_sizes, window_sizes, window_overlaps, chip_names))
    elif chip_names == [] and groups == []:
        parameter_combinations = list(
            itertools.product(bin_sizes, window_sizes, window_overlaps))

    # Base folder for storing results
    base_folder = os.path.join(settings.PATH_RESULTS_FOLDER, target_data_folder)

    folder_all_chips_list = []
    for combination in parameter_combinations:

        if chip_names != [] and groups != []:
            bin_size, window_size, window_overlap, chip_name, group = combination
            # Create a folder structure based on the parameter combination
            result_folder = os.path.join(base_folder, f"bin_{bin_size}",
                                         f"window_{window_size}", f"overlap_{window_overlap}", chip_name, group)
        elif chip_names != [] and groups == []:
            bin_size, window_size, window_overlap, chip_name = combination
            # Create a folder structure based on the parameter combination
            result_folder = os.path.join(base_folder, f"bin_{bin_size}",
                                         f"window_{window_size}", f"overlap_{window_overlap}", chip_name)
        elif chip_names == [] and groups == []:
            bin_size, window_size, window_overlap = combination
            # Create a folder structure based on the parameter combination
            result_folder = os.path.join(base_folder, f"bin_{bin_size}",
                                         f"window_{window_size}", f"overlap_{window_overlap}")



        folder_all_chips_list.append(result_folder)

    return folder_all_chips_list


# Old function: all paths are returned of the data being saved on the HD
def XXXget_all_paths(base_folder, common_folder_name='rec'):
    # base_folder: Base folder where the results are stored

    # get all chip paths
    folder_all_chips_list = []
    for root, dirs, files in os.walk(base_folder):

        # Extract parameters from the folder path
        parameters = root.split(os.sep)[1:]  # Adjust the index based on your folder structure

        # Check if the folder matches the specified parameters
        try:
            index = parameters.index(next(s for s in parameters if common_folder_name in s))
            index += 1
        except:
            index = None
        else:
            if len(parameters) == index:
                folder_all_chips_list.append(root)

    # in case there are duplets, make the paths unique. here already unique
    folder_all_chips_list = list(set(folder_all_chips_list))
    folder_all_chips_list = sorted(folder_all_chips_list)
    return folder_all_chips_list