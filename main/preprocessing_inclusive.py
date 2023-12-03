import torch 
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from models import *
from tools import *
from tqdm.auto import tqdm
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

# Device Agnostic Code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}\n")

# Define Data Paths and File names
DATA_DIR = Path("../data")
REC_FILE_NAME = "incl_rec.npy"
GEN_FILE_NAME = "incl_gen.npy"

# Load Data Files
file_incl_rec = np.load(DATA_DIR / REC_FILE_NAME)
file_incl_gen = np.load(DATA_DIR / GEN_FILE_NAME)

# Percentage of Data to be used
PERCENT = 1

# Merge Generated Data with Reconstructed Data
file_incl = np.r_[file_incl_rec[:int(file_incl_rec.shape[0]*PERCENT)], file_incl_gen[:int(file_incl_gen.shape[0]*PERCENT)]]
np.random.shuffle(file_incl)

# Print the shape of the data
print(f"The shape of the data is: {file_incl.shape}")

# Calculate the Pearson Correlation Coefficient
corr_matrix = np.corrcoef(file_incl, rowvar=False)

# Create a heatmap using Seaborn
variables = ['Xb', 'Y', 'Q2', 'Trig', 'PVz', 'PVx', 'PVy', 'Mom_mu', 'Mom_mup', 'dxdz_mup', 'dydz_mup']
plt.figure(figsize=(16, 12))
sns.set(font_scale=1)  # Adjust the font size if needed
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f",
            xticklabels=variables, yticklabels=variables)
plt.title("Pearson Correlation Matrix for Muons")

MODEL_PATH = Path("../graphs")
MODEL_PATH.mkdir(parents=True,
                 exist_ok=True)
MODEL_NAME = "inclusive_correlation_matrix.png"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Save the plot as an image (e.g., PNG)
plt.savefig(MODEL_SAVE_PATH, dpi=300)

# Find the indices of the most correlated variables
corr_matrix = np.abs(corr_matrix)
np.fill_diagonal(corr_matrix, 0)
corr_matrix = np.triu(corr_matrix)
corr_indices = np.where(corr_matrix > 0.9)
corr_indices = [(variables[i], variables[j]) for i, j in zip(corr_indices[0], corr_indices[1])]
print(f"The most correlated variables are: {corr_indices}")

# Remove the most correlated variables
for i, j in corr_indices:
    file_incl = np.delete(file_incl, variables.index(j), axis=1)
print(f"The shape of the preprocessed data is: {file_incl.shape}")

# Rewrite a new file with the preprocessed data
FILE_NAME = 'incl_preprocessed'
np.save(DATA_DIR / FILE_NAME, file_incl)