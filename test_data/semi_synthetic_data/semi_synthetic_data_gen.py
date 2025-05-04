#%% Import packages
import numpy as np
import pandas as pd
import cv2 as cv
from keras.datasets import mnist
semi_syn_data_dir = "E:/Happiness_source/PhD/UoB/projects/Mechanism Learning/code/dataset/simu_data/semi_synthetic_data/"
raw_data_dir = semi_syn_data_dir + "raw_data/"
frontdoor_data_dir = semi_syn_data_dir + "frontdoor_data/"
(train_X, train_y), (test_X, test_y) = mnist.load_data()
#%% Manipulate data
data_2 = train_X[train_y == 2].reshape(-1, 28*28)
data_6 = train_X[train_y == 6].reshape(-1, 28*28)
data = np.vstack((data_2, data_6))
# Load image data
# data_2 = pd.read_csv(raw_data_dir + "data_digit2.csv", header=None)
# data_6 = pd.read_csv(raw_data_dir + "data_digit6.csv", header=None)
# data = np.vstack((data_2.to_numpy(), data_6.to_numpy()))
labels = np.vstack((2*np.ones((data_2.shape[0], 1)), 6 * np.ones((data_6.shape[0], 1))))
N = data.shape[0]
digits = [2,6]
Ndigits = len(digits)
Nw, Nh = 28, 28
# U->Y strength (confounding)
quy = [0.80, 0.80, 0.50]

# Confounding background brightness adjustment
bright = 100

# Y->Z strength (primary effect)
qyz = [0.95, 0.95, 0.95]

Ne = 3  # Number of environments
rng_randn = np.random.default_rng(102)
rng_rand = np.random.default_rng(1)

# Randomly permute image data
perm = rng_rand.permutation(N)
data_perm = data[perm, :]
labels_perm = labels[perm].flatten()

# Select input image data as the basis for each environment
Nie = round(N / Ne)
ie = []
e = 0
i = 0
while i < N:
    il = np.arange(i, min(i + Nie, N))
    ie.append(il)
    i += Nie
    e += 1

U = {}
Y = {}
Z = {}
X = {}

for e_idx in range(Ne):
    # P(Z|Y)
    pz_y = np.array([
        [qyz[e_idx], 1 - qyz[e_idx]],
        [1 - qyz[e_idx], qyz[e_idx]]
    ])  # Shape (2,2)

    # Draw (continuous) confounder variable U
    M = int(round(len(ie[e_idx]) * 0.95))
    usig = 5
    umu = 0
    U[e_idx] = rng_randn.normal(loc=umu, scale=usig, size=M)

    # Draw (binary) target classification variable Y
    q = np.log((1 - quy[e_idx]) / quy[e_idx])
    # P(Y=2 | U) = 1 / (1 + exp(q * U))
    pyy = 1 / (1 + np.exp(q * U[e_idx]))
    u_rand = rng_rand.uniform(size=M)
    Y[e_idx] = (u_rand <= pyy).astype(int) + 1  # Y=1 or 2

    # Draw (binary) mediator variable Z
    u_rand_z = rng_rand.uniform(size=M)
    # Adjust indexing for Y (1 or 2)
    p = pz_y[1, Y[e_idx]-1]  # pz_y[y_class-1, :]
    Z[e_idx] = (u_rand_z <= p).astype(int) + 1  # Z=1 or 2

    # "Draw" image data variable X selected by Z, modified by U
    X[e_idx] = np.empty((M, Nw * Nh), dtype=np.float64)
    idx = {}
    for k, digit in enumerate(digits):
        idx[k] = np.intersect1d(np.where(labels_perm == digit)[0], ie[e_idx], assume_unique=True)
    j = np.ones(Ndigits, dtype=int)  # 1-based index
    for n in range(M):
        z = Z[e_idx][n] - 1  # 0 or 1
        digit_idx = z
        if j[digit_idx] > len(idx[digit_idx]):
            # Wrap around if not enough samples
            current_j = (j[digit_idx] - 1) % len(idx[digit_idx])
        else:
            current_j = j[digit_idx] - 1
        i_sample = idx[digit_idx][current_j]
        j[digit_idx] += 1
        # Modify brightness based on U
        x = data_perm[i_sample, :].astype(np.float64) + (0.5 * np.tanh(0.2 * U[e_idx][n]) + 0.5) * bright
        x = np.clip(x, 0, 255)
        X[e_idx][n, :] = x
            
#%% Save data
X_train_conf = pd.DataFrame(X[0])
Y_train_conf = pd.DataFrame(Y[0])
Z_train_conf = pd.DataFrame(Z[0])
X_train_conf.to_csv(frontdoor_data_dir + "X_train_conf.csv", header=None, index=None)
Y_train_conf.to_csv(frontdoor_data_dir + "Y_train_conf.csv", header=None, index=None)
Z_train_conf.to_csv(frontdoor_data_dir + "Z_train_conf.csv", header=None, index=None)

X_test_conf = pd.DataFrame(X[1])
Y_test_conf = pd.DataFrame(Y[1])
Z_test_conf = pd.DataFrame(Z[1])
X_test_conf.to_csv(frontdoor_data_dir + "X_test_conf.csv", header=None, index=None)
Y_test_conf.to_csv(frontdoor_data_dir + "Y_test_conf.csv", header=None, index=None)
Z_test_conf.to_csv(frontdoor_data_dir + "Z_test_conf.csv", header=None, index=None)

X_test_unconf = pd.DataFrame(X[2])
Y_test_unconf = pd.DataFrame(Y[2])
Z_test_unconf = pd.DataFrame(Z[2])
X_test_unconf.to_csv(frontdoor_data_dir + "X_test_unconf.csv", header=None, index=None)
Y_test_unconf.to_csv(frontdoor_data_dir + "Y_test_unconf.csv", header=None, index=None)
Z_test_unconf.to_csv(frontdoor_data_dir + "Z_test_unconf.csv", header=None, index=None)

