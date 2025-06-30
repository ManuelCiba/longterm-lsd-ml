# This script calcuates features from the synchrony curves such as peaks, AUC ect.

import matplotlib.pyplot as plt
import itertools  # generate all parameter combinations  for parameter study
import numpy as np
from lib.data_handler import hd
import settings
import os
import quantities as pq
import warnings
import pandas as pd
import re
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit
 
SOURCE_PATH = os.path.join(settings.PATH_RESULTS_FOLDER, settings.FOLDER_NAME_feature_synchrony)
TARGET_PATH = os.path.join(settings.PATH_RESULTS_FOLDER, settings.FOLDER_NAME_feature_synchrony_stats)

def calculate_curve_features_and_save(source_path, target_path):
    # load the synchrony curve file:
    df_curve = hd.load_csv_as_df(source_path)

    # calculate features from the synchrony curve
    df_features = _calculate_curve_features(df_curve)

    # save the features to a csv file
    hd.save_df_as_csv(df_features, target_path)

def _extract_time_scales_from_filename(df_curve):
    # Extract column names
    columns = df_curve.columns

    # Use a regular expression to extract the time scales
    time_scales = []
    for col in columns:
        match = re.search(r"Spike-contrast \(([\d.]+) s\)", col)
        if match:
            time_scales.append(float(match.group(1)))

    # Convert the list to a numpy array
    time_scales = np.array(time_scales)
    # flatten the array to ensure it is 1D
    time_scales = time_scales.flatten()

    return time_scales

def _calculate_curve_features(df_curve):

    # Extract time scales from the column names
    x = _extract_time_scales_from_filename(df_curve)

    # transform dataframe to numpy array
    y = df_curve.to_numpy()

    # Ensure y is a 1D array
    y = y.flatten()

    # calculate easy statistics
    mean_value = np.mean(y)
    std_value = np.std(y)
    min_value = np.min(y)
    max_value = np.max(y)
    max_value_bin = x[np.argmax(y)]
    median_value = np.median(y)

    # get synchrony for different time scales
    timescales = np.array([100, 10, 1, 0.1, 0.01])   # in seconds
    # Find the indices of the closest values
    indices = [np.abs(x - target).argmin() for target in timescales]
    # Extract the corresponding values from y
    synchrony_values = y[indices]

    #p1_x, p1_y, p1_w, p2_x, p2_y, p2_w = analyze_peaks_by_fitting_two_gaussians(x, y)
    #p1_x, p1_y, p2_x, p2_y = analyze_peaks_by_fitting_polynomial(x, y)
    #p1_x, p1_y, p2_x, p2_y = analyze_peaks_by_fitting_polinomial(x, y)

    # create a DataFrame to store the features
    df_features = pd.DataFrame({
        's_mean': [mean_value],
        's_std': [std_value],
        's_min': [min_value],
        's_max': [max_value],
        's_max_bin': [max_value_bin],
        's_median': [median_value],
        's_100': [synchrony_values[0]],
        's_10': [synchrony_values[1]],
        's_1': [synchrony_values[2]],
        's_0.1': [synchrony_values[3]],
        's_0.01': [synchrony_values[4]],
    })

    return df_features

# Define a single Gaussian function
def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Define a sum of two Gaussians
def two_gaussians(x, amp1, mu1, sigma1, amp2, mu2, sigma2):
    return gaussian(x, amp1, mu1, sigma1) + gaussian(x, amp2, mu2, sigma2)

def OLDanalyze_peaks_by_fitting_two_gaussians(x, y):
    # Initial guesses for two Gaussians
    initial_guess = [0.5, 20, 5, 0.5, 60, 10]  # [amp1, mu1, sigma1, amp2, mu2, sigma2]

    # Fit the curve
    params, _ = curve_fit(two_gaussians, x, y, p0=initial_guess)

    # Generate fitted curve
    fitted_y = two_gaussians(x, *params)

    # Extract parameters for each peak
    amp1, mu1, sigma1, amp2, mu2, sigma2 = params

    p1_x = mu1
    p1_y = amp1
    p1_w = sigma1  
    p2_x = mu2
    p2_y = amp2
    p2_w = sigma2

    return p1_x, p1_y, p1_w, p2_x, p2_y, p2_w

def analyze_peaks_by_fitting_polinomial(x, y):
    # Fit a polynomial
    degree = 4
    x_idx = np.arange(len(y))
    coeffs = np.polyfit(x_idx, y, deg=degree)
    polynomial = np.poly1d(coeffs)

    # Generate fitted curve
    #fitted_y = polynomial(x)

    # Find critical points (roots of the first derivative)
    first_derivative = polynomial.deriv()
    critical_points = np.roots(first_derivative)

    # Filter critical points to keep only those within the range of x
    valid_critical_points = critical_points[np.isreal(critical_points)]  # Only real roots
    valid_critical_points = valid_critical_points.astype(float)
    valid_critical_points = valid_critical_points[(valid_critical_points >= x.min()) & (valid_critical_points <= x.max())]

    # Evaluate the second derivative at critical points to classify peaks
    second_derivative = polynomial.deriv(2)
    peak_positions = valid_critical_points[second_derivative(valid_critical_points) < 0]  # Local maxima

    # Get peak heights
    peak_heights = polynomial(peak_positions)

    # Sort peaks by height
    sorted_indices = np.argsort(peak_heights)[::-1]
    peak_positions = peak_positions[sorted_indices]
    peak_heights = peak_heights[sorted_indices]

    p1_x = peak_positions[0]
    p1_y = peak_heights[0]
    p2_x = peak_positions[1] if len(peak_positions) > 1 else 0  # Handle case with less than 2 peaks 
    p2_y = peak_heights[1] if len(peak_heights) > 1 else 0  # Handle case with less than 2 peaks

    # Map peak positions back to original x values
    p1_x = x[np.round(p1_x).astype(int)] if p1_x >= 0 else 0
    p2_x = x[np.round(p2_x).astype(int)] if p2_x >= 0 else 0

    return p1_x, p1_y, p2_x, p2_y

def analyze_peaks_by_fitting_polynomial(x, y):
    """
    Analyze peaks in a curve by fitting a polynomial using indices for the fit 
    and mapping peak positions back to the original x values.

    Parameters:
    x (array-like): Original x-axis values.
    y (array-like): y-axis values of the curve.

    Returns:
    p1_x, p1_y, p2_x, p2_y: Positions (x) and heights (y) of the two highest peaks.
    """
    import numpy as np

    # Use indices for the polynomial fitting
    indices = np.arange(len(y))

    # Fit a polynomial using the indices
    degree = 4
    coeffs = np.polyfit(indices, y, deg=degree)
    polynomial = np.poly1d(coeffs)

    # Generate fitted curve
    fitted_y = polynomial(indices)

    # Find critical points (roots of the first derivative)
    first_derivative = polynomial.deriv()
    critical_points = np.roots(first_derivative)

    # Filter critical points to keep only those within the range of indices
    valid_critical_points = critical_points[np.isreal(critical_points)]  # Only real roots
    valid_critical_points = valid_critical_points.astype(float)
    valid_critical_points = valid_critical_points[
        (valid_critical_points >= indices.min()) & (valid_critical_points <= indices.max())
    ]

    # Evaluate the second derivative at critical points to classify peaks
    second_derivative = polynomial.deriv(2)
    peak_positions = valid_critical_points[second_derivative(valid_critical_points) < 0]  # Local maxima

    # Get peak heights
    peak_heights = polynomial(peak_positions)

    # Sort peaks by height
    sorted_indices = np.argsort(peak_heights)[::-1]
    peak_positions = peak_positions[sorted_indices]
    peak_heights = peak_heights[sorted_indices]

    # Map peak positions to original x values
    peak_positions_original = x[np.round(peak_positions).astype(int)]

    # Extract the two most prominent peaks
    if len(peak_positions) >= 2:
        p1_x, p1_y = peak_positions_original[0], peak_heights[0]
        p2_x, p2_y = peak_positions_original[1], peak_heights[1]
    elif len(peak_positions) == 1:
        p1_x, p1_y = peak_positions_original[0], peak_heights[0]
        p2_x, p2_y = 0, 0  # No second peak
    else:
        p1_x, p1_y, p2_x, p2_y = 0, 0, 0, 0  # No peaks found

    return p1_x, p1_y, p2_x, p2_y

if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    # define parameter
    bin_sizes = settings.BIN_SIZES
    window_sizes = settings.WINDOW_SIZES
    window_overlaps = settings.WINDOW_OVERLAPS

    # get all chip names
    chip_names = settings.WELLS_SHAM + settings.WELLS_DRUG  # Combine control and DRUG wells

    # Generate all combinations of parameters
    parameter_combinations = list(itertools.product(bin_sizes, window_sizes, window_overlaps))

    # loop through each combination of parameters
    for combination in parameter_combinations:
        bin_size, window_size, window_overlap = combination

        # Create a folder structure based on the parameter combination
        current_path = os.path.join(SOURCE_PATH, f"bin_{bin_size}",
                                     f"window_{window_size}", f"overlap_{window_overlap}")
  
        # get all directories in the current path (=chip names)
        chip_folders = [entry for entry in os.listdir(current_path) if os.path.isdir(os.path.join(current_path, entry))]    
        # sort the chip folders
        chip_folders.sort()

        # Loop through each chip folder
        for chip_folder in chip_folders:

            # Define the full path to the chip folder
            chip_folder_path = os.path.join(current_path, chip_folder)

            # List all directories in the current path
            rec_folders = [entry for entry in os.listdir(chip_folder_path) if os.path.isdir(os.path.join(chip_folder_path, entry))]

            # Loop through each folder (=recordings) and process the files
            for rec_folder in rec_folders:
                # Define the full path to the folder
                rec_folder_path = os.path.join(chip_folder_path, rec_folder)
                result_path = rec_folder_path.replace(SOURCE_PATH, TARGET_PATH)


                # List all .pkl files in the folder
                files = [f for f in os.listdir(rec_folder_path) if f.endswith('.csv')]

                # Process each .pkl file (= spike train file) and
                # calculate snychrony
                for file in files:
                    source_path = os.path.join(rec_folder_path, file)
                    target_path = os.path.join(result_path, file)

                    # Test if target file already exists
                    if False: #os.path.exists(target_path):
                        print(f"Already processed: {target_path}")
                        continue
                    else:
                        # Calculate synchrony and save results
                        print(f"Calculating synchrony curve stats: {target_path}")
                        calculate_curve_features_and_save(source_path, target_path)