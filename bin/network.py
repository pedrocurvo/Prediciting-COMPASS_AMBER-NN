import numpy as np
import tensorflow as tf
import math
from tensorflow import keras
from keras import mixed_precision
from math import sqrt
import matplotlib.pyplot as plt
import torch
from torch import nn

from keras.layers import Layer, Dense, Conv1D, Dropout , Input
from keras.models import Sequential
from keras.utils import to_categorical, get_custom_objects


x_bins = [0.004, 0.010, 0.020, 0.030, 0.040, 0.060, 0.100, 0.140, 0.180, 0.400]
y_bins = [0.10, 0.15, 0.20, 0.30, 0.50, 0.70]

file_incl_rec = np.load('data/incl_rec.npy')
file_incl_gen = np.load('data/incl_gen.npy')

matrix_r = np.zeros((len(x_bins), len(y_bins)))
matrix_g = np.zeros((len(x_bins), len(y_bins)))

file_inclu = np.r_[file_incl_rec[:], file_incl_gen[:]]
np.random.shuffle(file_inclu)
print(file_inclu.shape)
x = 11
x_train = file_inclu[:,0:x]
y_train = file_inclu[:,-1]
x_train = x_train.astype('float32')

print(x_train.shape)
print(y_train.shape)
print(f"X_train[0] : {x_train[0]}")
print(f"Y_train[0] : {y_train[0]}")

mean = x_train.mean(axis=0, keepdims=True)
stdev = x_train.std(axis=0, keepdims=True)
print(mean, stdev)
x_train_one = (x_train - mean) / stdev



#-----------------------------------Generated Incl Histogram-----------------------------------#
# ndarrays 
xb_array = file_incl_gen[:,0]
y_array = file_incl_gen[:,1]

# Define the range of the bins for the histograms
x_bins = [0.004, 0.010, 0.020, 0.030, 0.040, 0.060, 0.100, 0.140, 0.180, 0.400]
y_bins = [0.10, 0.15, 0.20, 0.30, 0.50, 0.70]

# Calculate the 2D histogram using np.histogram2d()
hist_gen, x_edges, y_edges = np.histogram2d(xb_array, y_array, bins=[x_bins, y_bins])

# Create a meshgrid for the bin edges
X, Y = np.meshgrid(x_edges, y_edges)

# Plot the 2D histogram
plt.pcolormesh(X, Y, hist_gen.T, cmap='plasma')
plt.colorbar(label='Counts')

plt.xlabel('Xb')
plt.ylabel('Y')
plt.yscale('log')
plt.xscale('log')

plt.title('Generated Incl Histogram')

#plt.show()

#-----------------------------------Reconstructed Incl Histogram-----------------------------------#
# ndarrays
xb_array = file_incl_rec[:,0]
y_array = file_incl_rec[:,1]

# Define the range of the bins for the histograms
x_bins = [0.004, 0.010, 0.020, 0.030, 0.040, 0.060, 0.100, 0.140, 0.180, 0.400]
y_bins = [0.10, 0.15, 0.20, 0.30, 0.50, 0.70]

# Calculate the 2D histogram using np.histogram2d()
hist_rec, x_edges, y_edges = np.histogram2d(xb_array, y_array, bins=[x_bins, y_bins])

# Create a meshgrid for the bin edges
X, Y = np.meshgrid(x_edges, y_edges)

# Plot the 2D histogram
plt.pcolormesh(X, Y, hist_rec.T, cmap='plasma')
plt.colorbar(label='Counts')

plt.xlabel('Xb')
plt.ylabel('Y')
plt.yscale('log')
plt.xscale('log')
plt.title('Reconstructed Incl Histogram')

#plt.show()

#-----------------------------------R+G Histogram-----------------------------------#
# ndarrays
xb_array = file_inclu[:,0]
y_array = file_inclu[:,1]

# Define the range of the bins for the histograms
x_bins = [0.004, 0.010, 0.020, 0.030, 0.040, 0.060, 0.100, 0.140, 0.180, 0.400]
y_bins = [0.10, 0.15, 0.20, 0.30, 0.50, 0.70]

# Calculate the 2D histogram using np.histogram2d()
hist_r_g, x_edges, y_edges = np.histogram2d(xb_array, y_array, bins=[x_bins, y_bins])

# Create a meshgrid for the bin edges
X, Y = np.meshgrid(x_edges, y_edges)

# Plot the 2D histogram
plt.pcolormesh(X, Y, hist_r_g.T, cmap='plasma')
plt.colorbar(label='Counts')

plt.xlabel('Xb')
plt.ylabel('Y')
plt.yscale('log')
plt.xscale('log')
plt.title('R+G Histogram')

#plt.show()

def wrapped_custom_metric(hist_gen, hist_rec, file_inclu):
    def metric2(y_true, y_pred):
        print("y_pred", y_pred[:,0])
        if y_pred.shape[0] == None:
            return 0
        return y_pred.shape[0]
    def metric(y_true, y_pred):
        # ndarrays
        print("y_pred", y_pred)
        xb_array = file_inclu[:,0]
        y_array = file_inclu[:,1]

        # Define the range of the bins for the histograms
        x_bins = [0.004, 0.010, 0.020, 0.030, 0.040, 0.060, 0.100, 0.140, 0.180, 0.400]
        y_bins = [0.10, 0.15, 0.20, 0.30, 0.50, 0.70]

        # Calculate the 2D histogram using np.histogram2d()
        hist_nnd, x_edges, y_edges = np.histogram2d(xb_array, y_array, bins=[x_bins, y_bins], weights=y_pred)

        hist_nnd_array = np.array(hist_nnd)
        r = np.array(hist_rec)
        g = np.array(hist_gen)
        mean = hist_nnd_array / (r + g)
        new_array = r / g - mean / (1 - mean)
        denominator = np.sqrt(np.sqrt(r) ** 2 + r * r * np.sqrt(g) ** 2 / g ** 2) / g
        new_array = new_array * new_array / denominator / denominator
        new_array = np.nan_to_num(new_array)
        chi_squared = np.sum(new_array)
        return chi_squared
    return metric2

model = Sequential()
model.add(Dense(100, input_dim=x, activation='swish'))
model.add(Dense(30, activation='swish'))
model.add(Dense(1, activation='sigmoid', dtype='float32'))
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(x_train_one, y_train, epochs=1, batch_size=100000, verbose=1, validation_split=0.2)
train = model.predict(x_train_one, batch_size=10000)
print(train[0:10])
# train = train / (1 - train)

print(train.shape)

# Define the range of the bins for the histograms
matrix = np.zeros((len(x_bins), len(y_bins)))
matrix_counts = np.zeros((len(x_bins), len(y_bins)))
#-----------------------------------NND Histogram-----------------------------------#
# ndarrays
xb_array = file_inclu[:,0]
y_array = file_inclu[:,1]

# Define the range of the bins for the histograms
x_bins = [0.004, 0.010, 0.020, 0.030, 0.040, 0.060, 0.100, 0.140, 0.180, 0.400]
y_bins = [0.10, 0.15, 0.20, 0.30, 0.50, 0.70]

# Calculate the 2D histogram using np.histogram2d()
hist_nnd, x_edges, y_edges = np.histogram2d(xb_array, y_array, bins=[x_bins, y_bins], weights=train[:,0])

# Create a meshgrid for the bin edges
X, Y = np.meshgrid(x_edges, y_edges)

# Plot the 2D histogram
plt.pcolormesh(X, Y, hist_nnd.T, cmap='plasma')
plt.colorbar(label='Counts')

plt.xlabel('Xb')
plt.ylabel('Y')
plt.yscale('log')
plt.xscale('log')
plt.title('NND Histogram')

#plt.show()
hist_nnd_array = np.array(hist_nnd)

#-----------------------------------chi square-----------------------------------#

# Define a wrapper function
def wrapped_custom_metric(hist_gen, hist_rec, file_inclu):
    def metric(y_true, y_pred):
        # ndarrays
        xb_array = file_inclu[:,0]
        y_array = file_inclu[:,1]

        # Define the range of the bins for the histograms
        x_bins = [0.004, 0.010, 0.020, 0.030, 0.040, 0.060, 0.100, 0.140, 0.180, 0.400]
        y_bins = [0.10, 0.15, 0.20, 0.30, 0.50, 0.70]

        # Calculate the 2D histogram using np.histogram2d()
        hist_nnd, x_edges, y_edges = np.histogram2d(xb_array, y_array, bins=[x_bins, y_bins], weights=y_pred[:,0])

        hist_nnd_array = np.array(hist_nnd)
        r = np.array(hist_rec)
        g = np.array(hist_gen)
        mean = hist_nnd_array / (r + g)
        new_array = r / g - mean / (1 - mean)
        denominator = np.sqrt(np.sqrt(r) ** 2 + r * r * np.sqrt(g) ** 2 / g ** 2) / g
        new_array = new_array * new_array / denominator / denominator
        new_array = np.nan_to_num(new_array)
        chi_squared = np.sum(new_array)
        return chi_squared
    return metric


chi_squared = 0
new_var = train[train[:, 0] > 0.5 , 0]
print(new_var.shape)

hist_nnd_array = np.array(hist_nnd)
hist_gen_array = np.array(hist_gen)
hist_rec_array = np.array(hist_rec)
r = hist_rec_array
g = hist_gen_array
mean = hist_nnd_array / (r + g)

new_array = r / g - mean / (1 - mean)
denominator = np.sqrt(np.sqrt(r) ** 2 + r * r * np.sqrt(g) ** 2 / g ** 2) / g
new_array = new_array * new_array / denominator / denominator
new_array = np.nan_to_num(new_array)
print(f"Chi_Squared per bin: \n {new_array} \n")
print(f"Error per bin: \n {denominator} \n")
print(f"R/G per bin: \n {r / g} \n")
print(f"NND per bin: \n {mean / (1 - mean)} \n")
print(f"R per bin: \n {r} \n")
print(f"G per bin: \n {g} \n")
chi_squared = np.sum(new_array)
print(f'Chi Squared: {chi_squared}')

# Let's Try PyTorch
#-----------------------------------PyTorch-----------------------------------#
# Setup device-agnostic code 
if torch.cuda.is_available():
    device = "cuda" # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = "mps" # Apple GPU
else:
    device = "cpu" # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available

print(f"Using device: {device}")
per = 0.3
X_train = torch.tensor(x_train_one[:int(len(x_train_one) * per)], dtype=torch.float32)
Y_train = torch.tensor(y_train[:int(len(x_train_one) * per)], dtype=torch.float32)
#xb_array = xb_array[:800000]
#y_array = y_array[:800000]
# Split data 
train_split = int(0.8 * len(X_train)) # 80% of data
# Separate out test data
X_test = X_train[train_split:]
y_test = Y_train[train_split:]

# Update training data to exclude test data
X_train = X_train[:train_split]
y_train = Y_train[:train_split]

print(len(X_train), len(y_train), len(X_test), len(y_test))

# Define model
class CustomModel(nn.Module):
    def __init__(self, input_dim):
        super(CustomModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 50, dtype=torch.float32)
        self.swish = nn.SiLU()  # Swish activation function
        self.layer2 = nn.Linear(50, 25)
        self.layer3 = nn.Linear(25, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.swish(self.layer1(x))
        x = self.swish(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x

model = CustomModel(input_dim=X_train.shape[1])

# Check the model current device
print(next(model.parameters()).device)
# Set the model to use the target device 
model.to(device)
# Check the model current device
print(next(model.parameters()).device)

# Define loss function
def custom_loss(y_pred, y_true, file_inclu, file_incl_rec, file_incl_gen):
    xb_array = file_inclu[:,0]
    y_array = file_inclu[:,1]
    print("y_pred_shape", y_pred.shape)
    per = 0.3 * 0.8
    print("xb_size", int(len(xb_array) * per))
    copy = y_pred[:,0].cpu().detach().numpy().copy()
    hist_nnd, x_edges, y_edges = np.histogram2d(xb_array[:int(len(xb_array) * per)], y_array[:int(len(y_array) * per)], bins=[x_bins, y_bins], weights=copy)
    xb_array = file_incl_rec[:,0]
    y_array = file_incl_rec[:,1]
    hist_rec, x_edges, y_edges = np.histogram2d(xb_array[:int(len(xb_array) * per)], y_array[:int(len(y_array) * per)], bins=[x_bins, y_bins])
    xb_array = file_incl_gen[:,0]
    y_array = file_incl_gen[:,1]
    hist_gen, x_edges, y_edges = np.histogram2d(xb_array[:int(len(xb_array) * per)], y_array[:int(len(y_array) * per)], bins=[x_bins, y_bins])

    chi_squared = 0

    hist_nnd_array = np.array(hist_nnd)
    hist_gen_array = np.array(hist_gen)
    hist_rec_array = np.array(hist_rec)
    r = hist_rec_array
    g = hist_gen_array
    mean = hist_nnd_array / (r + g)

    new_array = r / g - mean / (1 - mean)
    denominator = np.sqrt(np.sqrt(r) ** 2 + r * r * np.sqrt(g) ** 2 / g ** 2) / g
    new_array = new_array * new_array / denominator / denominator
    chi_squared = new_array
    chi_squared = torch.tensor(chi_squared, dtype=torch.float32)
    return chi_squared

loss_fn = nn.BCELoss()  # Binary Cross-Entropy loss
optimizer = torch.optim.Adam(params= model.parameters(),  lr=1e-2)


epochs = 1
# Put data on the target device (device agnostic code for data)
X_train, y_train = X_train.to(device), y_train.to(device).unsqueeze(1)
X_test, y_test = X_test.to(device), y_test.to(device).unsqueeze(1)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

for epoch in range(epochs):
    model.train()

    # 1. Forward Pass
    y_pred = model(X_train)

    # 2. Compute loss
    loss = loss_fn(y_pred, y_train)

    # 3. Optimizer Zero Grad
    optimizer.zero_grad()

    # 4. Backward Propagation
    loss.backward()

    # 5. Optimizer Step
    optimizer.step()

    ### Testing 
    model.eval() # turn off dropout and batch normalization
    with torch.inference_mode():
        test_pred = model(X_test)
        test_loss = loss_fn(test_pred, y_test)
    
    # Print out what's happening 
    if epoch % 1 == 0:
        print(f'Epoch: {epoch} | Loss: {loss.item():.3f} | Test loss: {test_loss.item():.3f}')


# Making and evaluating predictions 
# Turn the model into evaluation mode 

model.eval()
# Make predictions
with torch.inference_mode():
    train = model(torch.tensor(x_train_one[:int(len(x_train_one) * per)], dtype=torch.float32).to(device))
print(train.shape)
print(len(xb_array), len(y_array), len(train[:,0]))
xb_array = file_inclu[:,0]
y_array = file_inclu[:,1]
hist_nnd, x_edges, y_edges = np.histogram2d(xb_array[:int(len(xb_array) * per)], y_array[:int(len(y_array) * per)], bins=[x_bins, y_bins], weights=train[:,0].cpu())
xb_array = file_incl_rec[:,0]
y_array = file_incl_rec[:,1]
hist_rec, x_edges, y_edges = np.histogram2d(xb_array[:int(len(xb_array) * per)], y_array[:int(len(y_array) * per)], bins=[x_bins, y_bins])
xb_array = file_incl_gen[:,0]
y_array = file_incl_gen[:,1]
hist_gen, x_edges, y_edges = np.histogram2d(xb_array[:int(len(xb_array) * per)], y_array[:int(len(y_array) * per)], bins=[x_bins, y_bins])

chi_squared = 0

hist_nnd_array = np.array(hist_nnd)
hist_gen_array = np.array(hist_gen)
hist_rec_array = np.array(hist_rec)
r = hist_rec_array
g = hist_gen_array
mean = hist_nnd_array / (r + g)

new_array = r / g - mean / (1 - mean)
denominator = np.sqrt(np.sqrt(r) ** 2 + r * r * np.sqrt(g) ** 2 / g ** 2) / g
new_array = new_array * new_array / denominator / denominator
new_array = np.nan_to_num(new_array)
print(f"Chi_Squared per bin: \n {new_array} \n")
print(f"Error per bin: \n {denominator} \n")
print(f"R/G per bin: \n {r / g} \n")
print(f"NND per bin: \n {mean / (1 - mean)} \n")
print(f"R per bin: \n {r} \n")
print(f"G per bin: \n {g} \n")
chi_squared = np.sum(new_array)
print(f'Chi Squared: {chi_squared}')