from torch import nn 
from models import *
from pathlib import Path
import numpy as np
import torch
from tools import *

# Load Data Files
DATA_DIR = Path("../data")
FILE_NAME_REC = "hadr_rec1.npy"
FILE_NAME_GEN = "hadr_gen1.npy"

# Set up device
device = "cuda" if torch.cuda.is_available() else "cpu"

file_hadr_rec = np.load(DATA_DIR / FILE_NAME_REC)
file_hadr_gen = np.load(DATA_DIR / FILE_NAME_GEN)

file_hadr = np.r_[file_hadr_rec, file_hadr_gen]

# Elimitate Correlated Variables
file_hadr = np.delete(file_hadr, 9, axis=1)

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
model.load_state_dict(torch.load(f'../NN_hadr/{380}.pth'))
model.eval()
with torch.inference_mode():
    # 1. Forward pass
    test_logits = model(X).squeeze()
    test_pred = torch.sigmoid(test_logits)
    # Round 
    test_pred = torch.round(test_pred)

            


    