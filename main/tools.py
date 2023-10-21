import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from timeit import default_timer as timer

def plt_hist2d(xb_array, y_array, x_bins, y_bins, title, xlabel, ylabel, weights=None, log=True, density=False):

    # Calculate the 2D histogram using np.histogram2d()
    hist_gen, x_edges, y_edges = np.histogram2d(xb_array, y_array, bins=[x_bins, y_bins], weights=weights, density=density)

    # Create a meshgrid for the bin edges
    X, Y = np.meshgrid(x_edges, y_edges)

    # Plot the 2D histogram
    plt.pcolormesh(X, Y, hist_gen.T, cmap='plasma')
    plt.colorbar(label='Counts')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if log:
        plt.yscale('log')
        plt.xscale('log')
    plt.title(title)

    plt.plot()

def plt_hist(X, title, xlabel, ylabel, weights=None, log=True, bins=100, color='blue'):
    plt.hist(X, weights=weights, bins=bins, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if log:
        plt.yscale('log')
        plt.xscale('log')
    plt.plot()

# Calculate accuracy - out of 100 examples, what percentage did the model get correct?
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().float().item()
    return (correct / len(y_pred)) * 100

# Chi Squared Function
def chi_squared(y_true, y_pred, X_test, hadron=False):
    # ndarrays
    xb_array = X_test[:,0]
    y_array = X_test[:,1]

    # Define the range of the bins for the histograms
    x_bins = [0.004, 0.010, 0.020, 0.030, 0.040, 0.060, 0.100, 0.140, 0.180, 0.400]
    y_bins = [0.10, 0.15, 0.20, 0.30, 0.50, 0.70]

    # Calculate the 2D histogram using np.histogram2d()
    hist_nnd, x_edges, y_edges = np.histogram2d(xb_array, y_array, bins=[x_bins, y_bins], weights=y_pred)

    # Concatenate X_test and y_true
    X_test_y_true = np.c_[X_test, y_true]
    last_col = X_test_y_true[:,-1]

    r = X_test_y_true[last_col == 1]
    g = X_test_y_true[last_col == 0]

    r, x_edges, y_edges = np.histogram2d(r[:,0], r[:,1], bins=[x_bins, y_bins])
    g, x_edges, y_edges = np.histogram2d(g[:,0], g[:,1], bins=[x_bins, y_bins])

    nnd = np.array(hist_nnd)
    g = np.where(g == 0, 1, g)
    denominator = np.sqrt(np.sqrt(r) ** 2 + r * r * np.sqrt(g) ** 2 / g ** 2) / g
    denominator = np.where(denominator == 0, 1, denominator)
    alternative_chi_squared = (r * r + g * r - nnd * (r + g)) / (g * r + g * g - nnd) / denominator
    alternative_chi_squared = alternative_chi_squared * alternative_chi_squared
    return np.sum(np.nan_to_num(alternative_chi_squared))

def chi_squared_3d(y_true, y_pred, X_test):
    xb_array = X_test[:,0]
    y_array = X_test[:,1]
    z_array = X_test[:,2]

    # ndarrays
    xb_array = X_test[:,0]
    y_array = X_test[:,1]

    # Define the range of the bins for the histograms
    x_bins = [0.004, 0.010, 0.020, 0.030, 0.040, 0.060, 0.100, 0.140, 0.180, 0.400]
    y_bins = [0.10, 0.15, 0.20, 0.30, 0.50, 0.70]
    z_bins = [0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.85]

    hist_nnd = np.histogramdd((xb_array, y_array, z_array), bins=[x_bins, y_bins, z_bins], weights=y_pred)
    # Concatenate X_test and y_true
    X_test_y_true = np.c_[X_test, y_true]
    last_col = X_test_y_true[:,-1]

    r = X_test_y_true[last_col == 1]
    g = X_test_y_true[last_col == 0]

    hist_rec = np.histogramdd((r[:,0], r[:,1], r[:,2]), bins=[x_bins, y_bins, z_bins])
    hist_gen = np.histogramdd((g[:,0], g[:,1], g[:,2]), bins=[x_bins, y_bins, z_bins])

    nnd = torch.from_numpy(hist_nnd[0])
    r = torch.from_numpy(hist_rec[0])
    g = torch.from_numpy(hist_gen[0])

    g_plus = torch.clone(g)
    g_plus_r = g_plus + r
    g_plus_r = np.where(g_plus_r == 0, 1, g_plus_r)
    g = torch.from_numpy(np.where(g == 0, 1, g))
    denominator = np.sqrt(np.sqrt(r) ** 2 + r * r * np.sqrt(g) ** 2 / g ** 2) / g
    denominator = np.where(denominator == 0, 1, denominator)
    alternative_chi_squared = (r * r + g * r - nnd * (r + g)) / (g * r + g * g - nnd) / denominator
    alternative_chi_squared = alternative_chi_squared * alternative_chi_squared
    return np.sum(np.nan_to_num(alternative_chi_squared))



def print_train_time(start: float,
                     end: float,
                     device: torch.device = None):
    """ Prints difference between start and end time."""
    total_time = end - start
    print(f"Training time on {device}: {total_time:.3f} seconds")
    return total_time