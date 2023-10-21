from torch import nn 
from models import *
from pathlib import Path
import numpy as np
import torch
from tools import *

# Define if its Hadron or Inclusive
PREFIX = 'net'
IMAGE_DIR = Path("../graphs")

# Load Data Files
DATA_DIR = Path("../data")
FILE_NAME = "incl_preprocessed.npy"

# Set up device
device = "cuda" if torch.cuda.is_available() else "cpu"

file_incl = np.load(DATA_DIR / FILE_NAME)

# Split Data into Target and Features
X = torch.from_numpy(file_incl[:,0:(file_incl.shape[1]-1)]).type(torch.float32)
y = torch.from_numpy(file_incl[:,-1]).type(torch.float32)

# Normalize Data
mean = torch.mean(X, axis=0, keepdims=True)
stdev = torch.std(X, axis=0, keepdims=True)
X = (X - mean) / stdev

# remove Nans
X.nan_to_num(0)
y.nan_to_num(0)

model = ModelV2(input_dim=X.shape[1], hidden_dim=64).to(device)

# Load Model
model.load_state_dict(torch.load(f'../NN_incl/{28}.pth'))
model.eval()
with torch.inference_mode():
    # 1. Forward pass
    test_logits = model(X).squeeze()
    test_pred = torch.sigmoid(test_logits)
    # Round 
    #test_pred = torch.round(abs(test_pred - i))

# Plot 2D Histograms
plt.figure(figsize=(16, 9))

# Define the range of the bins for the histograms
x_bins = [0.004, 0.010, 0.020, 0.030, 0.040, 0.060, 0.100, 0.140, 0.180, 0.400]
y_bins = [0.10, 0.15, 0.20, 0.30, 0.50, 0.70]

#-----------------------------------Generated Incl Histogram-----------------------------------#
weights = test_pred.cpu().detach().numpy()
hist_nnd, x_edges, y_edges = np.histogram2d(X[:,0], X[:,1], bins=[x_bins, y_bins], weights=weights)
hist_true, x_edges, y_edges = np.histogram2d(X[:,0], X[:,1], bins=[x_bins, y_bins], weights=y)
new_hist = hist_nnd / hist_true
print(new_hist)


print(f'Mean for Set One: {new_hist.mean()}')
            


    