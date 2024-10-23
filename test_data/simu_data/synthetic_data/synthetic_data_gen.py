#%% Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
syn_data_dir = "E:/Happiness_source/PhD/UoB/projects/Mechanism Learning/code/dataset/simu_data/synthetic_data/"

#%% Simu Data 1: frontdoor_discY_contZ_contX_discU
np.random.seed(1)
testcase_dir = "frontdoor_discY_contZ_contX_discU/"
n_samples_confounded_train = 5000
n_samples_confounded_test = 1000
n_samples_unconfounded_test = 1000

qyz1 = 0.6
qz1x1 = 0.6
quy = 0.8
qux = 0.8

pu = np.array([0.75, 0.25])
py_u = np.array([[quy, 1 - quy], [1 - quy, quy]]) 
pyu = np.tile(pu, (2, 1)) * py_u 
py = np.sum(pyu, axis=1)

# Confounded training data
u_seed = np.random.rand(n_samples_confounded_train)
U_train_conf = np.where(u_seed <= pu[1], 2, 1)

y_seed = np.random.rand(n_samples_confounded_train)
p = py_u[1, U_train_conf - 1]
Y_train_conf = np.where(y_seed <= p, 2, 1)

muz = np.zeros((n_samples_confounded_train, 1)) 
i = (Y_train_conf == 2)
muz[i, 0] += 2 * qyz1
muz[~i, 0] -= 2 * qyz1

Z_train_conf = muz + np.random.randn(n_samples_confounded_train, 1)

mux = np.zeros((n_samples_confounded_train,2))
i = (Z_train_conf[:,0] >= 0)
mux[i, 0] += 3 * qz1x1
mux[~i, 0] -= 3 * qz1x1
i = (U_train_conf == 2)
mux[i, 1] += 3*qux
mux[~i, 1] -= 3*qux

X_train_conf = mux + np.random.randn(n_samples_confounded_train, 2)

# Confounded test data
u_seed = np.random.rand(n_samples_confounded_test)
U_test_conf = np.where(u_seed <= pu[1], 2, 1)

y_seed = np.random.rand(n_samples_confounded_test)
p = py_u[1, U_test_conf - 1]
Y_test_conf = np.where(y_seed <= p, 2, 1)

muz = np.zeros((n_samples_confounded_test, 1)) 
i = (Y_test_conf == 2)
muz[i, 0] += 2 * qyz1
muz[~i, 0] -= 2 * qyz1

Z_test_conf = muz + np.random.randn(n_samples_confounded_test, 1)

mux = np.zeros((n_samples_confounded_test,2))
i = (Z_test_conf[:,0] >= 0)
mux[i, 0] += 3 * qz1x1
mux[~i, 0] -= 3 * qz1x1
i = (U_test_conf == 2)
mux[i, 1] += 3*qux
mux[~i, 1] -= 3*qux

X_test_conf = mux + np.random.randn(n_samples_confounded_test, 2)

# Unconfounded test data
u_seed = np.random.rand(n_samples_unconfounded_test)
U_uncf_test = np.where(u_seed <= pu[1], 2, 1)

y_seed = np.random.rand(n_samples_unconfounded_test)
Y_uncf_test = np.where(y_seed <= py[1], 1, 2)

muz = np.zeros((n_samples_unconfounded_test, 1)) 
i = (Y_uncf_test == 2)
muz[i, 0] += 2 * qyz1
muz[~i, 0] -= 2 * qyz1

Z_uncf_test = muz + np.random.randn(n_samples_unconfounded_test, 1)

mux = np.zeros((n_samples_unconfounded_test,2))
i = (Z_uncf_test[:,0] >= 0)
mux[i, 0] += 3 * qz1x1
mux[~i, 0] -= 3 * qz1x1
i = (U_uncf_test == 2)
mux[i, 1] += 3*qux
mux[~i, 1] -= 3*qux

X_uncf_test = mux + np.random.randn(n_samples_unconfounded_test, 2)

# Save data
pd.DataFrame(U_train_conf).to_csv(syn_data_dir + testcase_dir + "U_train_conf.csv", index = False)
pd.DataFrame(X_train_conf).to_csv(syn_data_dir + testcase_dir + "X_train_conf.csv", index = False)
pd.DataFrame(Y_train_conf).to_csv(syn_data_dir + testcase_dir + "Y_train_conf.csv", index = False)
pd.DataFrame(Z_train_conf).to_csv(syn_data_dir + testcase_dir + "Z_train_conf.csv", index = False)

pd.DataFrame(U_test_conf).to_csv(syn_data_dir + testcase_dir + "U_test_conf.csv", index = False)
pd.DataFrame(X_test_conf).to_csv(syn_data_dir + testcase_dir + "X_test_conf.csv", index = False)
pd.DataFrame(Y_test_conf).to_csv(syn_data_dir + testcase_dir + "Y_test_conf.csv", index = False)
pd.DataFrame(Z_test_conf).to_csv(syn_data_dir + testcase_dir + "Z_test_conf.csv", index = False)


pd.DataFrame(U_uncf_test).to_csv(syn_data_dir + testcase_dir + "U_test_unconf.csv", index = False)
pd.DataFrame(X_uncf_test).to_csv(syn_data_dir + testcase_dir + "X_test_unconf.csv", index = False)
pd.DataFrame(Y_uncf_test).to_csv(syn_data_dir + testcase_dir + "Y_test_unconf.csv", index = False)
pd.DataFrame(Z_uncf_test).to_csv(syn_data_dir + testcase_dir + "Z_test_unconf.csv", index = False)

#%% Simu Data 2: frontdoor_ContY_contZ_contX_contU

testcase_dir = "frontdoor_contY_contZ_contX_contU/"
n_samples_confounded_train = 5000
n_samples_confounded_test = 1000
n_samples_unconfounded_test = 1000
np.random.seed(42)

a0 = 2.0
a1 = 1.0
b = 1.5
c = 5.0
d = 1.0

# Confounded training data
U_train_conf = np.random.normal(loc=0, scale=1, size=n_samples_confounded_train)
# Y = a0 + a1 * U + noise_Y
noise_Y = np.random.normal(loc=0, scale=1, size=n_samples_confounded_train)
Y_train_conf = (a0 + a1 * U_train_conf) + noise_Y
# Z = b * Y + noise_Z
noise_Z = np.random.normal(loc=0, scale=1, size=n_samples_confounded_train)
Z_train_conf = b * Y_train_conf + noise_Z
# X = c * U + d * Z + noise_X
noise_X = np.random.normal(loc=0, scale=1, size=n_samples_confounded_train)
X_train_conf = c * U_train_conf + d * Z_train_conf + noise_X

# Confounded test data
U_test_conf = np.random.normal(loc=0, scale=1, size=n_samples_confounded_test)
# Y = a0 + a1 * U + noise_Y
noise_Y = np.random.normal(loc=0, scale=1, size=n_samples_confounded_test)
Y_test_conf = (a0 + a1 * U_test_conf) + noise_Y
# Z = b * Y + noise_Z
noise_Z = np.random.normal(loc=0, scale=1, size=n_samples_confounded_test)
Z_test_conf = b * Y_test_conf + noise_Z
# X = c * U + d * Z + noise_X
noise_X = np.random.normal(loc=0, scale=1, size=n_samples_confounded_test)
X_test_conf = c * U_test_conf + d * Z_test_conf + noise_X

# Unconfounded test data
U_test_unconf = np.random.normal(loc=0, scale=1, size=n_samples_unconfounded_test)
# Y = a0 + a1 * U + noise_Y
noise_Y = np.random.normal(loc=0, scale=1, size=n_samples_unconfounded_test)
Y_test_unconf = (a0 + a1 * U_test_unconf) + noise_Y
# Z = b * Y + noise_Z
noise_Z = np.random.normal(loc=0, scale=1, size=n_samples_unconfounded_test)
Z_test_unconf = b * Y_test_unconf + noise_Z
# X = d * Z + noise_X
noise_X = np.random.normal(loc=0, scale=1, size=n_samples_unconfounded_test)
X_test_unconf = d * Z_test_unconf + noise_X

pd.DataFrame(X_train_conf).to_csv(syn_data_dir + testcase_dir + "X_train_conf.csv", index = False)
pd.DataFrame(Y_train_conf).to_csv(syn_data_dir + testcase_dir + "Y_train_conf.csv", index = False)
pd.DataFrame(U_train_conf).to_csv(syn_data_dir + testcase_dir + "U_train_conf.csv", index = False)
pd.DataFrame(Z_train_conf).to_csv(syn_data_dir + testcase_dir + "Z_train_conf.csv", index = False)

pd.DataFrame(X_test_conf).to_csv(syn_data_dir + testcase_dir + "X_test_conf.csv", index = False)
pd.DataFrame(Y_test_conf).to_csv(syn_data_dir + testcase_dir + "Y_test_conf.csv", index = False)
pd.DataFrame(U_test_conf).to_csv(syn_data_dir + testcase_dir + "U_test_conf.csv", index = False)
pd.DataFrame(Z_test_conf).to_csv(syn_data_dir + testcase_dir + "Z_test_conf.csv", index = False)

pd.DataFrame(X_test_unconf).to_csv(syn_data_dir + testcase_dir + "X_test_unconf.csv", index = False)
pd.DataFrame(Y_test_unconf).to_csv(syn_data_dir + testcase_dir + "Y_test_unconf.csv", index = False)
pd.DataFrame(U_test_unconf).to_csv(syn_data_dir + testcase_dir + "U_test_unconf.csv", index = False)
pd.DataFrame(Z_test_unconf).to_csv(syn_data_dir + testcase_dir + "Z_test_unconf.csv", index = False)


# %%
