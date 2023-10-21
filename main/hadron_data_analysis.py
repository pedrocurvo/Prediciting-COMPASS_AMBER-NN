import numpy as np
import matplotlib.pyplot as plt
from tools import plt_hist2d, plt_hist
from pathlib import Path

# Define if its Hadron or Inclusive
PREFIX = 'hadr'

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
titles = ['$X_b$', '$Y$', '$Z$', '$Q^2$', '$Trigger$', '$PV_z$', '$PV_x$', '$PV_y$', '$Beam Momentum$', '$Momentum \mu$', '$dx/dy \mu$', '$dy/dz \mu$', '$Charge$', '$Momentum$', '$dx/dy$', '$dy/dz$']

# Log Scales
logs = [True, False, True, True, False, False, True, True, True, True, True, True, False, True, True, True]

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