import numpy as np
import matplotlib.pyplot as plt
from tools import plt_hist2d, plt_hist
from pathlib import Path

# Define if its Hadron or Inclusive
PREFIX = 'incl'

# Define Directory Path
DATA_DIR = Path("../data")
FILE_REC = f"{PREFIX}_rec.npy"
FILE_GEN = f"{PREFIX}_gen.npy"
IMAGE_DIR = Path("../graphs")

# Load Files 
file_rec = np.load(DATA_DIR / FILE_REC)
file_gen = np.load(DATA_DIR / FILE_GEN)

# Check Shape of Files
print(f'Reconstructed Shape: {file_rec.shape}')
print(f'Generated Shape: {file_gen.shape}')

# Plots for Individual Variables Reconstructed and Generated
titles = ['$X_b$', '$Y$', '$Q^2$', '$Trigger$', '$PV_z$', '$PV_x$', '$PV_y$', '$Beam Momentum$', '$Momentum \mu$', '$dx/dz \mu$', '$dy/dz \mu$']

# Log Scales
logs = [True, False, True, False, False, True, True, True, True, True, True, False, True, True, True]

for i in range(len(titles)):
    plt.figure(figsize=(8, 6))
    plt.suptitle(titles[i], fontsize=16)

    # Histogram for Reconstructed Data Relative Frequency
    plt.hist(file_rec[:,i], bins=100, density=True, color='blue', alpha=0.5, label='Reconstructed', log=logs[i])
    plt.legend(loc='upper right')
    plt.xlabel(titles[i])
    plt.ylabel('Relative Frequency')

    # Histogram for Generated Data Relative Frequency
    plt.hist(file_gen[:,i], bins=100, density=True, color='orange', alpha=0.5, label='Generated', log=logs[i])
    plt.legend(loc='upper right')
    plt.xlabel(titles[i])
    plt.ylabel('Relative Frequency')

    # Save the plot as an image (e.g., PNG)
    name = titles[i].replace('/', '_').replace(' ', '_').replace('^', '').replace('$', '').replace('\\', '')
    IMAGE_NAME = f'{PREFIX}_{name}.png'
    IMAGE_SAVE_PATH = IMAGE_DIR / IMAGE_NAME
    plt.savefig(IMAGE_SAVE_PATH, dpi=300)
    print(f'Saving {IMAGE_NAME}...')


# 2D Histograms
plt.figure(figsize=(16, 9))

# Define the range of the bins for the histograms
x_bins = [0.004, 0.010, 0.020, 0.030, 0.040, 0.060, 0.100, 0.140, 0.180, 0.400]
y_bins = [0.10, 0.15, 0.20, 0.30, 0.50, 0.70]

#-----------------------------------Generated Incl Histogram-----------------------------------#
plt.subplot(2, 2, 1)
weights = file_gen[:,-1]
plt_hist2d(file_gen[:,0], file_gen[:,1], x_bins, y_bins, 'Generated Incl Histogram', 'Xb', 'Y')

# -----------------------------------2D Xb Y Rec Histogram-----------------------------------#
plt.subplot(2, 2, 2)
weights = file_rec[:,-1]
plt_hist2d(file_rec[:,0], file_rec[:,1], x_bins, y_bins, 'Reconstructed Incl Histogram', 'Xb', 'Y')

# Save the plot as an image (e.g., PNG)
IMAGE_NAME = f'{PREFIX}_xb_y.png'
IMAGE_SAVE_PATH = IMAGE_DIR / IMAGE_NAME
plt.savefig(IMAGE_SAVE_PATH, dpi=300)