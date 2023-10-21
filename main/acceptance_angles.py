import torch 
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from models import *
from tools import *
import matplotlib.pyplot as plt
from pathlib import Path

# Define if its Hadron or Inclusive
PREFIX = 'net'
IMAGE_DIR = Path("../graphs")

# Load Data Files
DATA_DIR = Path("../data")
HADR_FILE_NAME = "hadron_preprocessed.npy"
INCL_FILE_NAME = "incl_preprocessed.npy"
# Merge Generated Data with Reconstructed Data
file_hadr = np.load(DATA_DIR / HADR_FILE_NAME)
file_incl = np.load(DATA_DIR / INCL_FILE_NAME)

# Split Data into Target and Features
X_hadr = torch.from_numpy(file_hadr[:,0:(file_hadr.shape[1]-1)]).type(torch.float32)
y_hadr = torch.from_numpy(file_hadr[:,-1]).type(torch.float32)

X_incl = torch.from_numpy(file_incl[:,0:(file_incl.shape[1]-1)]).type(torch.float32)
y_incl = torch.from_numpy(file_incl[:,-1]).type(torch.float32)

# Normalize Data
mean = torch.mean(X_hadr, axis=0, keepdims=True)
stdev = torch.std(X_hadr, axis=0, keepdims=True)
X_hadr = (X_hadr - mean) / stdev

mean_incl = torch.mean(X_incl, axis=0, keepdims=True)
stdev_incl = torch.std(X_incl, axis=0, keepdims=True)
X_incl = (X_incl - mean_incl) / stdev_incl

# Create Model
model_hadr = ModelV2(input_dim=X_hadr.shape[1], hidden_dim=64)
model_incl = ModelV2(input_dim=X_incl.shape[1], hidden_dim=64)

model_hadr.load_state_dict(torch.load('../NN_hadr/317.pth'))
model_incl.load_state_dict(torch.load('../NN_incl/28.pth'))

model_hadr.eval()
model_incl.eval()
with torch.inference_mode():
    # 1. Forward pass
    test_logits_hadr = model_hadr(X_hadr).squeeze()
    test_pred_hadr = torch.sigmoid(test_logits_hadr) # No need to round, since we are looking for the NN acceptance
    test_logits_incl = model_incl(X_incl).squeeze()
    test_pred_incl = torch.sigmoid(test_logits_incl) # No need to round, since we are looking for the NN acceptance

    plt.figure(figsize=(10,7))
    plt.hist2d(X_hadr[:,-2].cpu(), X_hadr[:,-1].cpu(), bins=20000, weights=test_pred_hadr.cpu())
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.savefig(IMAGE_DIR / "hadron_angle.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10,7))
    plt.hist2d(X_incl[:,-2].cpu(), X_incl[:,-1].cpu(), bins=20000, weights=test_pred_incl.cpu())
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.savefig(IMAGE_DIR / "incl_angle.png", dpi=300)
    plt.close()