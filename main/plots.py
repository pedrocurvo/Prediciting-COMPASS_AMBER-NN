import numpy as np
import matplotlib.pyplot as plt
from tools import plt_hist2d, plt_hist

if __name__ == '__main__':
    # Load Files 
    file_incl_rec = np.load('data/incl_rec.npy')
    file_incl_gen = np.load('data/incl_gen.npy')

    # Percentage of Data to be used
    percent = 0.5

    print(file_incl_rec.shape)
    print(file_incl_gen.shape)
    file_incl_rec = file_incl_rec[:int(file_incl_rec.shape[0]*percent),:]
    file_incl_gen = file_incl_gen[:int(file_incl_gen.shape[0]*percent),:]
    print(file_incl_rec.shape)
    print(file_incl_gen.shape)

    file_incl = np.r_[file_incl_rec, file_incl_gen]
    print(file_incl.shape)
    np.random.shuffle(file_incl)

    # Weights 
    weights = file_incl[:,-1]
    print(weights[:10])

    ### 1D Histograms ###
    titles = ['Xb', 'Y', 'Q2', 'Trigger', 'PVz', 'PVx', 'PVy', 'Beam Momentum', 'Momentum u', 'dx/dy u', 'dy/dz u']
    logs = [True, False, True, True, False, False, True, True, True, True, True]
    for i in range(len(titles)):
        plt.figure(figsize=(13, 6))
        plt.suptitle(titles[i], fontsize=16)
        plt.subplot(1, 4, 3)
        plt_hist(file_incl[:,i], 'Incl', titles[i], 'Counts', weights=file_incl[:,-1], log=logs[i])
        plt.subplot(1, 4, 1)
        plt_hist(file_incl_gen[:,i], 'Gen', titles[i], 'Counts', weights=file_incl_gen[:,-1], log=logs[i])
        plt.subplot(1, 4, 2)
        plt_hist(file_incl_rec[:,i], 'Rec', titles[i], 'Counts', weights=file_incl_rec[:,-1], log=logs[i])
        plt.show()


    



    ### 2D Histograms ###
    plt.figure(figsize=(15, 8))

    # Define the range of the bins for the histograms
    x_bins = [0.004, 0.010, 0.020, 0.030, 0.040, 0.060, 0.100, 0.140, 0.180, 0.400]
    y_bins = [0.10, 0.15, 0.20, 0.30, 0.50, 0.70]

    #-----------------------------------Generated Incl Histogram-----------------------------------#
    plt.subplot(2, 2, 1)
    weights = file_incl_gen[:,-1]
    plt_hist2d(file_incl_gen[:,0], file_incl_gen[:,1], x_bins, y_bins, 'Generated Incl Histogram', 'Xb', 'Y')

    # -----------------------------------2D Xb Y Rec Histogram-----------------------------------#
    plt.subplot(2, 2, 2)
    weights = file_incl_rec[:,-1]
    plt_hist2d(file_incl_rec[:,0], file_incl_rec[:,1], x_bins, y_bins, 'Reconstructed Incl Histogram', 'Xb', 'Y', weights=weights)

    # -----------------------------------R+G Histogram-----------------------------------#
    plt.subplot(2, 2, 3)
    weights = file_incl[:,-1]
    plt_hist2d(file_incl[:,0], file_incl[:,1], x_bins, y_bins, 'R+G Histogram', 'Xb', 'Y', weights=weights)

    plt.show()


