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

# Save the plot as an image (e.g., PNG)
plt.savefig("correlation_matrix.png", dpi=300)

# Split Data into Target and Features
X = torch.from_numpy(file_hadr[:,0:(file_hadr.shape[1]-1)]).type(torch.float32)
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
# Loss Function
loss_fn = nn.BCEWithLogitsLoss()
# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Number of Epochs
EPOCHS = 10

# Put all data on target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)


# Create model directory path
MODEL_PATH = Path("models_hadr")
MODEL_PATH.mkdir(parents=True,
                 exist_ok=True)

#model.load_state_dict(torch.load('models_hadr/34.pth'))
model = ModelV2(input_dim=X_train.shape[1], hidden_dim=64).to(device)
model.load_state_dict(torch.load('models_hadr/434.pth'))
model.eval()
with torch.inference_mode():
    # 1. Forward pass
    test_logits = model(X).squeeze()
    test_pred = torch.sigmoid(test_logits) # No need to round, since we are looking for the NN acceptance

    plt.figure(figsize=(10,7))
    plt.hist2d(X[:,-1].cpu(), X[:,-2].cpu(), bins=15000, weights=test_pred.cpu())
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.show()


# Training Loop
for epoch in tqdm(range(EPOCHS)):
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
        print(f'Train Accuracy: {acc:.2f}% | Test Accuracy: {test_acc:.2f}% | Chi Squared: {chi_squared_3d(y_test, test_pred, X_test)}\n')

    # Save model with chi_squared as name 
    MODEL_NAME = f"{chi_squared_3d(y_test, test_pred, X_test):.0f}.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
    # Save the model state dict
    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj=model.state_dict(),
            f=MODEL_SAVE_PATH)
