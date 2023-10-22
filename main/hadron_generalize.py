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
FILE_NAME = "hadron_preprocessed.npy"

# Set up device
device = "cuda" if torch.cuda.is_available() else "cpu"

file_hadr = np.load(DATA_DIR / FILE_NAME)

# Split Data into Target and Features
X = torch.from_numpy(file_hadr[:,0:(file_hadr.shape[1]-1)]).type(torch.float32)
y = torch.from_numpy(file_hadr[:,-1]).type(torch.float32)

# Normalize Data
mean = torch.mean(X, axis=0, keepdims=True)
stdev = torch.std(X, axis=0, keepdims=True)
X = (X - mean) / stdev

# remove Nans
X.nan_to_num(0)
y.nan_to_num(0)

model = ModelV2(input_dim=X.shape[1], hidden_dim=64).to(device)

# Load Model
model.load_state_dict(torch.load(f'../NN_hadr/317.pth'))
model.eval()
with torch.inference_mode():
    # 1. Forward pass
    test_logits = model(X).squeeze()
    test_pred = torch.sigmoid(test_logits)
    # Round 
    #test_pred = torch.round(abs(test_pred))

# Plot 2D Histograms
plt.figure(figsize=(16, 9))

#-----------------------------------Generated Incl Histogram-----------------------------------#
weights = test_pred.cpu().detach().numpy()

# Define the range of the bins for the histograms
x_bins = [0.004, 0.010, 0.020, 0.030, 0.040, 0.060, 0.100, 0.140, 0.180, 0.400]
y_bins = [0.10, 0.15, 0.20, 0.30, 0.50, 0.70]
z_bins = [0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.85]

hist_nnd = np.histogramdd((X[:,0], X[:,1], X[:, 2]), bins=[x_bins, y_bins, z_bins], weights=weights)
hist_true = np.histogramdd((X[:,0], X[:,1], X[:, 2]), bins=[x_bins, y_bins, z_bins], weights=y)
np.where(hist_true[0] == 0, 1, hist_true[0])
np.nan_to_num(hist_nnd[0])
np.nan_to_num(hist_true[0])
new_hist = hist_nnd[0] / hist_true[0]
new_hist = np.nan_to_num(new_hist, posinf=1, neginf=1)

print(f'Mean for Set One: {new_hist.mean()}')


# Generalize for New Data
hadr_rec = np.load(DATA_DIR / "hadr_rec1.npy")
hadr_gen = np.load(DATA_DIR / "hadr_gen1.npy")
hadr = np.concatenate((hadr_rec, hadr_gen), axis=0)
hadr = np.delete(hadr, 9, axis=1)

# Split Data into Target and Features
X = torch.from_numpy(hadr[:,0:(hadr.shape[1]-1)]).type(torch.float32)
y = torch.from_numpy(hadr[:,-1]).type(torch.float32)

# Normalize Data
mean = torch.mean(X, axis=0, keepdims=True)
stdev = torch.std(X, axis=0, keepdims=True)
X = (X - mean) / stdev

# remove Nans
X.nan_to_num(0)
y.nan_to_num(0)

with torch.inference_mode():
    # 1. Forward pass
    test_logits = model(X).squeeze()
    test_pred = torch.sigmoid(test_logits)
    # Round 
    #test_pred = torch.round(abs(test_pred))

# Plot 2D Histograms
plt.figure(figsize=(16, 9))

weights = test_pred.cpu().detach().numpy()

# Define the range of the bins for the histograms
x_bins = [0.004, 0.010, 0.020, 0.030, 0.040, 0.060, 0.100, 0.140, 0.180, 0.400]
y_bins = [0.10, 0.15, 0.20, 0.30, 0.50, 0.70]
z_bins = [0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.85]

hist_nnd = np.histogramdd((X[:,0], X[:,1], X[:, 2]), bins=[x_bins, y_bins, z_bins], weights=weights)
hist_true = np.histogramdd((X[:,0], X[:,1], X[:, 2]), bins=[x_bins, y_bins, z_bins], weights=y)
np.where(hist_true[0] == 0, 1, hist_true[0])
np.nan_to_num(hist_nnd[0])
np.nan_to_num(hist_true[0])
new_hist = hist_nnd[0] / hist_true[0]
new_hist = np.nan_to_num(new_hist, posinf=1, neginf=1)

print(f'Mean for Set Two: {new_hist.mean()}')




            


    