import torch 
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from models import *
from tools import *
import matplotlib.pyplot as plt

# Percentage of Data to be used
PERCENT = 1

# Number of Epochs
EPOCHS = 15

# Device Agnostic Code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}\n")

# Load Data Files
file_hadr_rec = np.load('data/hadr_rec.npy')
file_hadr_gen = np.load('data/hadr_gen.npy')

# Merge Generated Data with Reconstructed Data
file_hadr = np.r_[file_hadr_rec[:int(file_hadr_rec.shape[0]*PERCENT)], file_hadr_gen[:int(file_hadr_gen.shape[0]*PERCENT)]]
np.random.shuffle(file_hadr)

# Split Data into Target and Features
X = torch.from_numpy(file_hadr[:,0:(file_hadr.shape[1]-1)]).type(torch.float32)
#X = torch.index_select(X, dim=1, index=torch.tensor([i for i in range(X.shape[1]) if i != 8])) # Remove Trig
y = torch.from_numpy(file_hadr[:,-1]).type(torch.float32)

# Normalize Data
mean = torch.mean(X, axis=0, keepdims=True)
stdev = torch.std(X, axis=0, keepdims=True)
X = (X - mean) / stdev

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2, # 20% of data for test set and 80% for training set
                                                    random_state=42
                                                    )

# Create Model
model = ModelV2(input_dim=X_train.shape[1], hidden_dim=64).to(device)

model.load_state_dict(torch.load('models_hadr/434.pth'))
model.eval()
with torch.inference_mode():
    # 1. Forward pass
    test_logits = model(X).squeeze()
    test_pred = torch.sigmoid(test_logits) # No need to round, since we are looking for the NN acceptance

    plt.figure(figsize=(10,7))
    plt.hist2d(X[:,-2].cpu(), X[:,-1].cpu(), bins=20000, weights=test_pred.cpu())
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.show()
    plt.savefig("acceptance.png", dpi=300)