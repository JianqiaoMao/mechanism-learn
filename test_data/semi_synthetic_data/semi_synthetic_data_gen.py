#%% Import packages
import numpy as np
import pandas as pd
from keras.datasets import mnist

def generate_data_for_subset(
    data_sub, labels_sub,
    digits,
    quy,        
    qyz,        
    bright,
    rng_rand_seed=1,
    rng_randn_seed=102,
    frac=0.95   
):

    rng_rand  = np.random.default_rng(rng_rand_seed)
    rng_randn = np.random.default_rng(rng_randn_seed)

    N_sub = data_sub.shape[0]
    Nw = Nh = 28
    Ndigits = len(digits)

    perm = rng_rand.permutation(N_sub)
    data_perm   = data_sub[perm, :]
    labels_perm = labels_sub[perm]

    M = int(round(N_sub * frac))  

    # Generate confounder U
    usig = 5
    umu = 0
    U = rng_randn.normal(loc=umu, scale=usig, size=M)

    # Draw (binary) target classification variable Y
    q = np.log((1 - quy) / quy)
    # P(Y=2 | U) = 1 / (1 + exp(q * U))
    pyy = 1 / (1 + np.exp(q * U))

    u_rand = rng_rand.uniform(size=M)
    Y = (u_rand <= pyy).astype(int) + 1    # 1 or 2

    # Generate (binary) mechanism variable Z
    pz_y = np.array([
        [qyz, 1 - qyz],
        [1 - qyz, qyz]
    ])
    u_rand_z = rng_rand.uniform(size=M)
    p = pz_y[1, Y-1]
    Z = (u_rand_z <= p).astype(int) + 1    # 1 or 2

    # Generate effect X
    X = np.empty((M, Nw * Nh), dtype=np.float64)

    idx = {}
    for k, digit in enumerate(digits):
        idx[k] = np.where(labels_perm == digit)[0]
    j = np.ones(Ndigits, dtype=int)  

    for n in range(M):
        z = Z[n] - 1   
        digit_idx = z  

        if j[digit_idx] > len(idx[digit_idx]):
            current_j = (j[digit_idx] - 1) % len(idx[digit_idx])
        else:
            current_j = j[digit_idx] - 1

        i_sample = idx[digit_idx][current_j]
        j[digit_idx] += 1

        x = data_perm[i_sample, :].astype(np.float64) \
            + (0.5 * np.tanh(0.2 * U[n]) + 0.5) * bright
        x = np.clip(x, 0, 255)
        X[n, :] = x

    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)
    Z = pd.DataFrame(Z)
    U = pd.DataFrame(U)
    
    return X, Y, Z, U

#%%
semi_syn_data_dir = ""
frontdoor_data_dir = semi_syn_data_dir 
(train_X, train_y), _ = mnist.load_data()
data_2 = train_X[train_y == 2].reshape(-1, 28*28)
data_6 = train_X[train_y == 6].reshape(-1, 28*28)
data = np.vstack((data_2, data_6))
labels = np.vstack(
    (2*np.ones((data_2.shape[0], 1)),
     6*np.ones((data_6.shape[0], 1)))
).flatten()

N = data.shape[0]
rng_split = np.random.default_rng(0)
perm_all = rng_split.permutation(N)

n_train_base = int(0.8 * N)
idx_train_base = perm_all[:n_train_base]
idx_test_base  = perm_all[n_train_base:]

data_train_base   = data[idx_train_base]
labels_train_base = labels[idx_train_base]
data_test_base    = data[idx_test_base]
labels_test_base  = labels[idx_test_base]

digits = [2, 6]
bright = 100

# Confounded train
X_conf_train,   Y_conf_train,   Z_conf_train,   U_conf_train = generate_data_for_subset(
    data_train_base, labels_train_base,
    digits,
    quy=0.80,
    qyz=0.95,
    bright=bright,
    rng_rand_seed=1,
    rng_randn_seed=101,
    frac=0.95      
)

# Unconfounded train
X_unconf_train, Y_unconf_train, Z_unconf_train, U_unconf_train = generate_data_for_subset(
    data_train_base, labels_train_base,
    digits,
    quy=0.5,
    qyz=0.95,
    bright=bright,
    rng_rand_seed=2,
    rng_randn_seed=202,
    frac=0.95
)

# Confounded test
X_conf_test,   Y_conf_test,   Z_conf_test,   U_conf_test = generate_data_for_subset(
    data_test_base, labels_test_base,
    digits,
    quy=0.80,
    qyz=0.95,
    bright=bright,
    rng_rand_seed=3,
    rng_randn_seed=303,
    frac=0.95
)

# Unconfounded test
X_unconf_test, Y_unconf_test, Z_unconf_test, U_unconf_test = generate_data_for_subset(
    data_test_base, labels_test_base,
    digits,
    quy=0.5,
    qyz=0.95,
    bright=bright,
    rng_rand_seed=4,
    rng_randn_seed=404,
    frac=0.95
)
            
#%% Save data
X_conf_train.to_csv(frontdoor_data_dir + "X_train_conf.csv", header=None, index=None)
Y_conf_train.to_csv(frontdoor_data_dir + "Y_train_conf.csv", header=None, index=None)
Z_conf_train.to_csv(frontdoor_data_dir + "Z_train_conf.csv", header=None, index=None)

X_unconf_train.to_csv(frontdoor_data_dir + "X_train_unconf.csv", header=None, index=None)
Y_unconf_train.to_csv(frontdoor_data_dir + "Y_train_unconf.csv", header=None, index=None)
Z_unconf_train.to_csv(frontdoor_data_dir + "Z_train_unconf.csv", header=None, index=None)   

X_conf_test.to_csv(frontdoor_data_dir + "X_test_conf.csv", header=None, index=None)
Y_conf_test.to_csv(frontdoor_data_dir + "Y_test_conf.csv", header=None, index=None)
Z_conf_test.to_csv(frontdoor_data_dir + "Z_test_conf.csv", header=None, index=None)

X_unconf_test.to_csv(frontdoor_data_dir + "X_test_unconf.csv", header=None, index=None)
Y_unconf_test.to_csv(frontdoor_data_dir + "Y_test_unconf.csv", header=None, index=None)
Z_unconf_test.to_csv(frontdoor_data_dir + "Z_test_unconf.csv", header=None, index=None)

# %%
