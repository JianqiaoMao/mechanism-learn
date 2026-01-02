#%% Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

def visualize_clas_data(X, Y, dataset_type, ax=None, show_plot=False):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    shuffle_idx = np.random.permutation(X.shape[0])
    X = X[shuffle_idx]
    Y = Y[shuffle_idx]
    
    scatter = ax.scatter(
        x = X[:,0], 
        y = X[:,1], 
        c = Y, 
        s = 7, 
        cmap = 'coolwarm', 
        alpha = 0.3,
        label = dataset_type
    )

    handles_scatter, labels_scatter = scatter.legend_elements(prop="colors")

    # ax.set_xlim(-6, 6)
    # ax.set_ylim(-7, 7)

    true_b = ax.plot([0, 0], [-6, 6], '-.', c="black", linewidth=2, label="True boundary")
    confounder = ax.plot([-6, 6], [0, 0], '-.', c="orange", linewidth=2, label="Confounder boundary")

    ax.legend(
        handles = handles_scatter + true_b + confounder,
        labels = ['Class 1', 'Class 2', 'True class boundary', 'Confounder boundary'],
        loc = 'lower right'
    )
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")

    if show_plot and ax is None:
        plt.show()

    return ax

#%% Synthetic classification

def data_gen_discY_contM_contV_contX_discU_contW(n_samples, 
                                                 qu1y, 
                                                 qu2y, 
                                                 qym,
                                                 qmv,
                                                 qu1w,
                                                 qmw,
                                                 qvw,
                                                 qu2x2,
                                                 qwx1,
                                                 qvx1,
                                                 p_u1,
                                                 p_u2):

    pu1 = np.array([1-p_u1, p_u1])
    pu2 = np.array([1-p_u2, p_u2])

    # Confounded training data
    u1_seed = np.random.rand(n_samples)
    U1 = np.where(u1_seed <= pu1[1], 1, -1)
    u2_seed = np.random.rand(n_samples)
    U2 = np.where(u2_seed <= pu2[1], 1, -1)
    
    # Y depends on U1 and U2
    eps_y = np.random.randn(n_samples)
    y_seed = qu1y * U1 + qu2y * U2 + eps_y
    Y = np.where(y_seed >= qu1y*(2*p_u1-1) + qu2y*(2*p_u2-1), 1, -1)
    
    # M depends on Y
    mum = np.zeros((n_samples, 1))
    i = (Y == 1)
    mum[i, 0] += qym
    mum[~i, 0] -= qym
    M = mum + np.random.randn(n_samples, 1)
    
    # V depends on M
    muv = np.zeros((n_samples, 1))
    i = (M[:,0] >= 0)
    muv[i, 0] += qmv
    muv[~i, 0] -= qmv
    V = muv + np.random.randn(n_samples, 1)
    
    # W depends on U1, M, V
    muw = np.zeros((n_samples, 1))
    i = (U1 == 1)
    muw[i, 0] += qu1w
    muw[~i, 0] -= qu1w
    i = (M[:,0] >= 0)
    muw[i, 0] += qmw
    muw[~i, 0] -= qmw
    i = (V[:,0] >= 0)
    muw[i, 0] += qvw
    muw[~i, 0] -= qvw
    W = muw + np.random.randn(n_samples, 1)
    
    # X depends on U2, W, V; X1 on W, V; X2 on U2
    mux = np.zeros((n_samples, 2))
    i = (W[:,0] >= 0)
    mux[i, 0] += qwx1
    mux[~i, 0] -= qwx1
    i = (V[:,0] >= 0)
    mux[i, 0] += qvx1
    mux[~i, 0] -= qvx1
    i = (U2 == 1)
    mux[i, 1] += qu2x2
    mux[~i, 1] -= qu2x2
    X = mux + np.random.randn(n_samples, 2)
    
    return X, Y, M, V, W

np.random.seed(42)
testcase_dir = r"syn_classification/"

n_samples_confounded_train = 50000
n_samples_unconfounded_train = 50000
n_samples_confounded_test = 5000
n_samples_unconfounded_test = 5000

X_train_conf, Y_train_conf, M_train_conf, V_train_conf, W_train_conf = \
    data_gen_discY_contM_contV_contX_discU_contW(
    n_samples = n_samples_confounded_train,
    qu1y = 2,
    qu2y = 1.5,
    qym = 1.0,
    qmv = 2.5,
    qu1w = 2,
    qmw = 1,
    qvw = 1,
    qu2x2 = 2,
    qwx1 = 1,
    qvx1 = 1,
    p_u1 = 0.7,
    p_u2 = 0.7
    )
    
X_test_conf, Y_test_conf, M_test_conf, V_test_conf, W_test_conf = \
data_gen_discY_contM_contV_contX_discU_contW(
    n_samples = n_samples_confounded_test,
    qu1y = 2,
    qu2y = 1.5,
    qym = 1.0,
    qmv = 2.5,
    qu1w = 2,
    qmw = 1,
    qvw = 1,
    qu2x2 = 2,
    qwx1 = 1,
    qvx1 = 1,
    p_u1 = 0.7,
    p_u2 = 0.7
)
    
X_train_uncf, Y_train_uncf, M_train_uncf, V_train_uncf, W_train_uncf = \
data_gen_discY_contM_contV_contX_discU_contW(
    n_samples = n_samples_unconfounded_train,
    qu1y = 0,
    qu2y = 0,
    qym = 1.0,
    qmv = 2.5,
    qu1w = 2,
    qmw = 1,
    qvw = 1,
    qu2x2 = 2,
    qwx1 = 1,
    qvx1 = 1,
    p_u1 = 0.7,
    p_u2 = 0.7
)

X_test_uncf, Y_test_uncf, M_test_uncf, V_test_uncf, W_test_uncf = \
data_gen_discY_contM_contV_contX_discU_contW(
    n_samples = n_samples_unconfounded_test,
    qu1y = 0,
    qu2y = 0,
    qym = 1.0,
    qmv = 2.5,
    qu1w = 2,
    qmw = 1,
    qvw = 1,
    qu2x2 = 2,
    qwx1 = 1,
    qvx1 = 1,
    p_u1 = 0.7,
    p_u2 = 0.7
)
fig, axes = plt.subplots(2, 2, figsize=(8, 7))
axes = axes.flatten()

visualize_clas_data(X_train_conf, Y_train_conf, "Confounded train set", ax=axes[0])
visualize_clas_data(X_test_conf,  Y_test_conf,  "Confounded test set",  ax=axes[1])
visualize_clas_data(X_train_uncf, Y_train_uncf, "Unconfounded train set", ax=axes[2])
visualize_clas_data(X_test_uncf,  Y_test_uncf,  "Unconfounded test set",  ax=axes[3])

plt.tight_layout()
plt.show()
#%%
pd.DataFrame(X_train_conf).to_csv(testcase_dir + "X_train_conf.csv", index = False)
pd.DataFrame(Y_train_conf).to_csv(testcase_dir + "Y_train_conf.csv", index = False)
pd.DataFrame(M_train_conf).to_csv(testcase_dir + "M_train_conf.csv", index = False)

pd.DataFrame(X_test_conf).to_csv(testcase_dir + "X_test_conf.csv", index = False)
pd.DataFrame(Y_test_conf).to_csv(testcase_dir + "Y_test_conf.csv", index = False)
pd.DataFrame(M_test_conf).to_csv(testcase_dir + "M_test_conf.csv", index = False)

pd.DataFrame(X_train_uncf).to_csv(testcase_dir + "X_train_unconf.csv", index = False)
pd.DataFrame(Y_train_uncf).to_csv(testcase_dir + "Y_train_unconf.csv", index = False)
pd.DataFrame(M_train_uncf).to_csv(testcase_dir + "M_train_unconf.csv", index = False)

pd.DataFrame(X_test_uncf).to_csv(testcase_dir + "X_test_unconf.csv", index = False)
pd.DataFrame(Y_test_uncf).to_csv(testcase_dir + "Y_test_unconf.csv", index = False)
pd.DataFrame(M_test_uncf).to_csv(testcase_dir + "M_test_unconf.csv", index = False)


#%% Synthetic regression
from scipy.stats import linregress
testcase_dir = r"syn_regression/"

n_samples_confounded_train = 50000
n_samples_unconfounded_train = 50000
n_samples_confounded_test = 10000
n_samples_unconfounded_test = 10000
np.random.seed(42)

def data_gen_contY_contZ_contX_contU(
    n_samples, 
    a0, 
    a1,    
    b, 
    c0, 
    c1,
    d0,
    sigma_y=1.0,
    sigma_z=1.0,
    sigma_x=1.0,
    seed=None,
):
    rng = np.random.default_rng(seed)
    
    # U ~ N(0,1)
    U = rng.normal(loc=0.0, scale=1.0, size=n_samples)

    # Y = a0 + a1 * U + eps_Y   
    noise_Y = rng.normal(loc=0.0, scale=sigma_y, size=n_samples)
    Y = a0 + a1 * U + noise_Y

    # Z = b * Y + eps_Z
    noise_Z = rng.normal(loc=0.0, scale=sigma_z, size=n_samples)
    Z = b * Y + noise_Z

    # X = d0 + c0 * U + c1 * Z + eps_X
    noise_X = rng.normal(loc=0.0, scale=sigma_x, size=n_samples)
    X = d0 + c0 * U + c1 * Z + noise_X
    
    return X, Y, Z, U

a0 = 2
a1 = 1
b  = -1.5
c0_conf = 6
c0_unconf = 0
c1 = -2
d0 = 0

sigma_y = 1.0
sigma_z = 1
sigma_x = 1.0

# confounded train and test data
X_train_conf, Y_train_conf, Z_train_conf, U_train_conf = data_gen_contY_contZ_contX_contU(
    n_samples=n_samples_confounded_train,
    a0=a0,
    a1=a1,
    b=b,
    c0=c0_conf,
    c1=c1,
    d0=d0,
    sigma_y=sigma_y,
    sigma_z=sigma_z,
    sigma_x=sigma_x,
    seed=0
)

X_test_conf, Y_test_conf, Z_test_conf, U_test_conf = data_gen_contY_contZ_contX_contU(
    n_samples=n_samples_confounded_test,
    a0=a0,
    a1=a1,
    b=b,
    c0=c0_conf,
    c1=c1,
    d0=d0,
    sigma_y=sigma_y,
    sigma_z=sigma_z,
    sigma_x=sigma_x,
    seed=1
)

# unconfounded train and test data
X_train_unconf, Y_train_unconf, Z_train_unconf, U_train_unconf = data_gen_contY_contZ_contX_contU(
    n_samples=n_samples_unconfounded_train,
    a0=a0,
    a1=a1,  
    b=b,
    c0=c0_unconf,
    c1=c1,
    d0=d0,
    sigma_y=sigma_y,
    sigma_z=sigma_z,
    sigma_x=sigma_x,
    seed=2
)

X_test_unconf, Y_test_unconf, Z_test_unconf, U_test_unconf = data_gen_contY_contZ_contX_contU(
    n_samples=n_samples_unconfounded_test,
    a0=a0,
    a1=a1,
    b=b,
    c0=c0_unconf,
    c1=c1,
    d0=d0,
    sigma_y=sigma_y,
    sigma_z=sigma_z,
    sigma_x=sigma_x,
    seed=3
)

# Visualization
X_range = np.linspace(np.min(X_test_conf), np.max(X_test_conf), 1000)
plt.figure(figsize=(8, 5))
# Confounded Dataset Scatter Plot
plt.scatter(X_test_conf, Y_test_conf, alpha=0.5, color='red', label='Confounded', s=10)
# Linear Regression for Confounded Dataset
slope_conf, intercept_conf, _, _, _ = linregress(X_test_conf, Y_test_conf)
plt.plot(X_range, intercept_conf + slope_conf * X_range, 'r--', label='Confounded Regression', linewidth=4)

# Unconfounded Dataset Scatter Plot
plt.scatter(X_test_unconf, Y_test_unconf, alpha=0.5, color='blue', label='Unconfounded', s=10)
# Linear Regression for Unconfounded Dataset
slope_unconf, intercept_unconf, _, _, _ = linregress(X_test_unconf, Y_test_unconf)
plt.plot(X_range, intercept_unconf + slope_unconf * X_range, 'b--', label='Unconfounded Regression', linewidth=4)
plt.legend()
plt.show()
#%

pd.DataFrame(X_train_conf).to_csv(testcase_dir + "X_train_conf.csv", index = False)
pd.DataFrame(Y_train_conf).to_csv(testcase_dir + "Y_train_conf.csv", index = False)
pd.DataFrame(U_train_conf).to_csv(testcase_dir + "U_train_conf.csv", index = False)
pd.DataFrame(Z_train_conf).to_csv(testcase_dir + "Z_train_conf.csv", index = False)

pd.DataFrame(X_test_conf).to_csv(testcase_dir + "X_test_conf.csv", index = False)
pd.DataFrame(Y_test_conf).to_csv(testcase_dir + "Y_test_conf.csv", index = False)
pd.DataFrame(U_test_conf).to_csv(testcase_dir + "U_test_conf.csv", index = False)
pd.DataFrame(Z_test_conf).to_csv(testcase_dir + "Z_test_conf.csv", index = False)

pd.DataFrame(X_train_unconf).to_csv(testcase_dir + "X_train_unconf.csv", index = False)
pd.DataFrame(Y_train_unconf).to_csv(testcase_dir + "Y_train_unconf.csv", index = False)
pd.DataFrame(U_train_unconf).to_csv(testcase_dir + "U_train_unconf.csv", index = False)
pd.DataFrame(Z_train_unconf).to_csv(testcase_dir + "Z_train_unconf.csv", index = False)

pd.DataFrame(X_test_unconf).to_csv(testcase_dir + "X_test_unconf.csv", index = False)
pd.DataFrame(Y_test_unconf).to_csv(testcase_dir + "Y_test_unconf.csv", index = False)
pd.DataFrame(U_test_unconf).to_csv(testcase_dir + "U_test_unconf.csv", index = False)
pd.DataFrame(Z_test_unconf).to_csv(testcase_dir + "Z_test_unconf.csv", index = False)

# %%
