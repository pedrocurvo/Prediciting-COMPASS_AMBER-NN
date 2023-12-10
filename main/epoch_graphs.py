import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Path to CSVs 
CSV_PATH = Path("../csv")

# Load CSVs
hadr_shallow = pd.read_csv(CSV_PATH / "nn_hadrons_metrics.csv")
hadr_deep = pd.read_csv(CSV_PATH / "nn_hadrons_metrics_deeper.csv")
incl_shallow = pd.read_csv(CSV_PATH / "nn_incl_metrics.csv")
incl_deep = pd.read_csv(CSV_PATH / "nn_incl_metrics_deeper.csv")

# Hadr Shallow
plt.figure(figsize=(9, 6))
for i in range(5):
    plt.plot(hadr_shallow["epoch"].iloc[i*10:(i+1)*10], hadr_shallow["chi_squared"].iloc[i*10:(i+1)*10], label="Training Session {}".format(i+1))

plt.xlabel('Epochs')
plt.ylabel(r'$\chi^2$')
plt.xticks(np.arange(0, 10, 1))
plt.legend()
plt.savefig("../graphs/nn_hadrons_metrics.png")
plt.show()

# Hadr Deep
plt.figure(figsize=(9, 6))
for i in range(5):
    plt.plot(hadr_deep["epoch"].iloc[i*10:(i+1)*10], hadr_deep["chi_squared"].iloc[i*10:(i+1)*10], label="Training Session {}".format(i+1))

plt.xlabel('Epochs')
plt.ylabel(r'$\chi^2$')
plt.xticks(np.arange(0, 10, 1))
plt.legend()
plt.savefig("../graphs/nn_hadrons_metrics_deeper.png")
plt.show()

# Incl Shallow
plt.figure(figsize=(9, 6))
for i in range(5):
    plt.plot(incl_shallow["epoch"].iloc[i*10:(i+1)*10], incl_shallow["chi_squared"].iloc[i*10:(i+1)*10], label="Training Session {}".format(i+1))

plt.xlabel('Epochs')
plt.ylabel(r'$\chi^2$')
plt.xticks(np.arange(0, 10, 1))
plt.legend()
plt.savefig("../graphs/nn_incl_metrics.png")
plt.show()

# Incl Deep
plt.figure(figsize=(9, 6))
for i in range(5):
    plt.plot(incl_deep["epoch"].iloc[i*10:(i+1)*10], incl_deep["chi_squared"].iloc[i*10:(i+1)*10], label="Training Session {}".format(i+1))

plt.xlabel('Epochs')
plt.ylabel(r'$\chi^2$')
plt.xticks(np.arange(0, 10, 1))
plt.legend()
plt.savefig("../graphs/nn_incl_metrics_deeper.png")
plt.show()

