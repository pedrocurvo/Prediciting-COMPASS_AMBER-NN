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

# Load Data Files
file_hadr_rec = np.load('data/hadr_rec.npy')
file_hadr_gen = np.load('data/hadr_gen.npy')

# Percentage of Data to be used
PERCENT = 1

# Merge Generated Data with Reconstructed Data
file_hadr = np.r_[file_hadr_rec[:int(file_hadr_rec.shape[0]*PERCENT)], file_hadr_gen[:int(file_hadr_gen.shape[0]*PERCENT)]]
np.random.shuffle(file_hadr)

# Calculate the Pearson Correlation Coefficient
corr_matrix = np.corrcoef(file_hadr[:-1], rowvar=False)

# Create a heatmap using Seaborn
variables = ['Xb', 'Y', 'Z', 'Q2', 'Trig', 'PVz', 'PVx', 'PVy', 'Mom_mu', 'Mom_mup', 'dxdz_mup', 'dydz_mup', 'q', 'mom', 'dxdz', 'dydz']
plt.figure(figsize=(16, 12))
sns.set(font_scale=1)  # Adjust the font size if needed
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f",
            xticklabels=variables, yticklabels=variables)
plt.title("Pearson Correlation Matrix")

MODEL_PATH = Path("graphs")
MODEL_PATH.mkdir(parents=True,
                 exist_ok=True)
MODEL_NAME = "hadron_correlation_matrix.png"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Save the plot as an image (e.g., PNG)
plt.savefig(MODEL_SAVE_PATH, dpi=300)