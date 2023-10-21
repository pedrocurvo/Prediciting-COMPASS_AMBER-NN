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

# Percentage of Data to be used
PERCENT = 1

# Number of Epochs
EPOCHS = 10

# Device Agnostic Code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}\n")

# Load Data Files
DATA_DIR = Path("../data")
FILE_NAME = "incl_preprocessed.npy"
file_incl = np.load(DATA_DIR / FILE_NAME)

# Split Data into Target and Features
X = torch.from_numpy(file_incl[:,0:(file_incl.shape[1]-1)]).type(torch.float32)
y = torch.from_numpy(file_incl[:,-1]).type(torch.float32)

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
model = ModelV1(input_dim=X_train.shape[1], hidden_dim=16).to(device)

# Loss Function
loss_fn = nn.BCEWithLogitsLoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# Put all data on target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)


# Create model directory path
MODEL_PATH = Path("../NN_incl_deeper")
MODEL_PATH.mkdir(parents=True,
                 exist_ok=True)


list_epoch = []
list_train_loss = []
list_test_loss = []
list_train_acc = []
list_test_acc = []
list_chi_squared = []
# Training Loop
for epoch in tqdm(range(EPOCHS)):
    list_epoch.append(epoch)
    # Training
    model.train()
    # 1. Forward pass
    y_logits = model(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    # 2. Calculate loss
    loss = loss_fn(y_logits,
                     y_train)
    acc = accuracy_fn(y_true=y_train,
                        y_pred=y_pred)
    
    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Backward pass
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    # Testing
    model.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = model(X_test).squeeze()
        test_pred = torch.sigmoid(test_logits) # No need to round, since we are looking for the NN acceptance

        # 2. Calculate loss
        test_loss = loss_fn(test_logits,
                            y_test)
        test_acc = accuracy_fn(y_true=y_test,
                               y_pred=torch.round(test_pred))
    
    # Print out what's happening
    if epoch % 1 == 0:
        print(f'Epoch: {epoch} | Train Loss: {loss.item():.3f} | Test Loss: {test_loss.item():.3f}')
        print(f'Train Accuracy: {acc:.2f}% | Test Accuracy: {test_acc:.2f}% | Chi Squared: {chi_squared(y_test, test_pred, X_test)}\n')
        list_train_loss.append(loss.item())
        list_test_loss.append(test_loss.item())
        list_train_acc.append(acc)
        list_test_acc.append(test_acc)
        list_chi_squared.append(chi_squared(y_test, test_pred, X_test))

    # Save model with chi_squared as name 
    MODEL_NAME = f"{chi_squared(y_test, test_pred, X_test):.0f}.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
    # Save the model state dict
    print(f"Saving model to: {MODEL_SAVE_PATH}")
    #torch.save(obj=model.state_dict(),
    #        f=MODEL_SAVE_PATH)

# Plot Loss
PLOT_PATH = Path('../graphs')
PLOT_PATH.mkdir(parents=True,
                exist_ok=True)
PLOT_NAME = "nn_incl_metrics_deeper.png"
PLOT_SAVE_PATH = PLOT_PATH / PLOT_NAME
plt.figure(figsize=(10,7))
plt.plot(list_epoch, list_train_loss, label='Train Loss')
plt.plot(list_epoch, list_test_loss, label='Test Loss')
plt.plot(list_epoch, list_train_acc, label='Train Accuracy')
plt.plot(list_epoch, list_test_acc, label='Test Accuracy')
plt.plot(list_epoch, list_chi_squared, label='Chi Squared')
plt.legend()
plt.xlabel('Epoch')
plt.savefig(PLOT_SAVE_PATH, dpi=300)
plt.show()