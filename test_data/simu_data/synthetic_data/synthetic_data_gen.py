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
%matplotlib qt5
# Visualization
fig,ax=plt.subplots(figsize = (8, 7))
scatter = ax.scatter(x= X_train_conf[:,0], y = X_train_conf[:,1], c = Y_train_conf, s = 10, cmap='coolwarm', alpha = 0.5,
                     label = "Confounded test set")
handles_scatter, labels_scatter = scatter.legend_elements(prop="colors")

plt.xlim(-6,6)
plt.ylim(-7,7)

true_b = plt.plot([0, 0], [-8, 8], '-.k', linewidth=2, label="True boundary")
confounder = plt.plot([-8,8], [0,0], '-.y', linewidth = 2, label = "Confounder boundary")

ax.legend(handles=handles_scatter+true_b+confounder, labels=['Class 1',
                                                             'Class 2',
                                                             'True class boundary',
                                                             'Confounder boundary'], loc='lower right')
ax.set_xlabel("X1")
ax.set_ylabel("X2")
plt.tight_layout()
plt.show()
#%% Simu Data 2: frontdoor_ContY_contZ_contX_contU

# testcase_dir = "frontdoor_contY_contZ_contX_contU/"
# n_samples_confounded_train = 5000
# n_samples_confounded_test = 1000
# n_samples_unconfounded_test = 1000

# np.random.seed(42)
# # Confounded training data
# U_train_conf = np.random.normal(2, 0.5, n_samples_confounded_train)
# Y_train_conf = np.random.normal(1 * U_train_conf + 1, 2, n_samples_confounded_train)
# Z_train_conf = np.random.normal(3 * Y_train_conf + 1, 1, n_samples_confounded_train)
# X_train_conf = np.random.normal(-10 * U_train_conf + 3 * Z_train_conf, 0.5, n_samples_confounded_train)

# # Confounded test data
# U_test_conf = np.random.normal(2, 0.5, n_samples_confounded_test)
# Y_test_conf = np.random.normal(1 * U_test_conf + 1, 2, n_samples_confounded_test)
# Z_test_conf = np.random.normal(3 * Y_test_conf + 1, 1, n_samples_confounded_test)
# X_test_conf = np.random.normal(-10 * U_test_conf + 3 * Z_test_conf, 0.5, n_samples_confounded_test)

# # Unconfounded dataset
# U_test_unconf = np.random.normal(2, 0.5, n_samples_unconfounded_test)
# Y_test_unconf = np.random.normal(np.mean(Y_test_conf), np.std(Y_test_conf), n_samples_unconfounded_test) 
# Z_test_unconf = np.random.normal(3 * Y_test_unconf, 1, n_samples_unconfounded_test)
# X_test_unconf = np.random.normal(-10 * U_test_conf +3 * Z_test_unconf, 0.5, n_samples_unconfounded_test)

# # Save data
# # pd.DataFrame(X_train_conf).to_csv(syn_data_dir + testcase_dir + "X_train_conf.csv", index = False)
# # pd.DataFrame(Y_train_conf).to_csv(syn_data_dir + testcase_dir + "Y_train_conf.csv", index = False)
# # pd.DataFrame(U_train_conf).to_csv(syn_data_dir + testcase_dir + "U_train_conf.csv", index = False)
# # pd.DataFrame(Z_train_conf).to_csv(syn_data_dir + testcase_dir + "Z_train_conf.csv", index = False)

# # pd.DataFrame(X_test_conf).to_csv(syn_data_dir + testcase_dir + "X_test_conf.csv", index = False)
# # pd.DataFrame(Y_test_conf).to_csv(syn_data_dir + testcase_dir + "Y_test_conf.csv", index = False)
# # pd.DataFrame(U_test_conf).to_csv(syn_data_dir + testcase_dir + "U_test_conf.csv", index = False)
# # pd.DataFrame(Z_test_conf).to_csv(syn_data_dir + testcase_dir + "Z_test_conf.csv", index = False)

# # pd.DataFrame(X_test_unconf).to_csv(syn_data_dir + testcase_dir + "X_test_unconf.csv", index = False)
# # pd.DataFrame(Y_test_unconf).to_csv(syn_data_dir + testcase_dir + "Y_test_unconf.csv", index = False)
# # pd.DataFrame(U_test_unconf).to_csv(syn_data_dir + testcase_dir + "U_test_unconf.csv", index = False)
# # pd.DataFrame(Z_test_unconf).to_csv(syn_data_dir + testcase_dir + "Z_test_unconf.csv", index = False)

# # Visualization
# X_range = np.linspace(np.min(X_train_conf), np.max(X_train_conf), 1000)
# plt.figure(figsize=(12, 6))
# # Confounded Dataset Scatter Plot
# plt.scatter(X_test_conf, Y_test_conf, alpha=0.5, color='red', label='Confounded', s=10)
# # Linear Regression for Confounded Dataset
# slope_conf, intercept_conf, _, _, _ = linregress(X_test_conf, Y_test_conf)
# plt.plot(X_range, intercept_conf + slope_conf * X_range, 'r--', label='Confounded Regression')

# # Unconfounded Dataset Scatter Plot
# plt.scatter(X_test_unconf, Y_test_unconf, alpha=0.5, color='blue', label='Unconfounded', s=10)
# # Linear Regression for Unconfounded Dataset
# slope_unconf, intercept_unconf, _, _, _ = linregress(X_test_unconf, Y_test_unconf)
# plt.plot(X_range, intercept_unconf + slope_unconf * X_range, 'b--', label='Unconfounded Regression')
# plt.legend()
# plt.show()

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

# 生成混杂变量 U
U_train_conf = np.random.normal(loc=0, scale=1, size=n_samples_confounded_train)
# 生成 Y，Y 依赖于 U
# 斜率 a 随 U 变化，例如 a = a0 + a1 * U
noise_Y = np.random.normal(loc=0, scale=1, size=n_samples_confounded_train)
Y_train_conf = (a0 + a1 * U_train_conf) + noise_Y
# 生成 Z，Z 依赖于 Y
noise_Z = np.random.normal(loc=0, scale=1, size=n_samples_confounded_train)
Z_train_conf = b * Y_train_conf + noise_Z
# 生成 X，X 依赖于 U 和 Z
noise_X = np.random.normal(loc=0, scale=1, size=n_samples_confounded_train)
X_train_conf = c * U_train_conf + d * Z_train_conf + noise_X


# 生成混杂变量 U
U_test_conf = np.random.normal(loc=0, scale=1, size=n_samples_confounded_test)
# 生成 Y，Y 依赖于 U
# 斜率 a 随 U 变化，例如 a = a0 + a1 * U
noise_Y = np.random.normal(loc=0, scale=1, size=n_samples_confounded_test)
Y_test_conf = (a0 + a1 * U_test_conf) + noise_Y
# 生成 Z，Z 依赖于 Y
noise_Z = np.random.normal(loc=0, scale=1, size=n_samples_confounded_test)
Z_test_conf = b * Y_test_conf + noise_Z
# 生成 X，X 依赖于 U 和 Z
noise_X = np.random.normal(loc=0, scale=1, size=n_samples_confounded_test)
X_test_conf = c * U_test_conf + d * Z_test_conf + noise_X


# 生成混杂变量 U
U_test_unconf = np.random.normal(loc=0, scale=1, size=n_samples_unconfounded_test)
# 生成 Y，Y 依赖于 U
# 斜率 a 随 U 变化，例如 a = a0
noise_Y = np.random.normal(loc=0, scale=1, size=n_samples_unconfounded_test)
Y_test_unconf = (a0 + a1 * U_test_unconf) + noise_Y
# 生成 Z，Z 依赖于 Y
noise_Z = np.random.normal(loc=0, scale=1, size=n_samples_unconfounded_test)
Z_test_unconf = b * Y_test_unconf + noise_Z
# 生成 X，X 依赖于 U 和 Z
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



# Visualization
X_range = np.linspace(np.min(X_test_conf), np.max(X_test_conf), 1000)
plt.figure(figsize=(12, 6))
# Confounded Dataset Scatter Plot
plt.scatter(X_test_conf, Y_test_conf, alpha=0.5, color='red', label='Confounded', s=10)
# Linear Regression for Confounded Dataset
slope_conf, intercept_conf, _, _, _ = linregress(X_test_conf, Y_test_conf)
plt.plot(X_range, intercept_conf + slope_conf * X_range, 'r--', label='Confounded Regression')

# Unconfounded Dataset Scatter Plot
plt.scatter(X_test_unconf, Y_test_unconf, alpha=0.5, color='blue', label='Unconfounded', s=10)
# Linear Regression for Unconfounded Dataset
slope_unconf, intercept_unconf, _, _, _ = linregress(X_test_unconf, Y_test_unconf)
plt.plot(X_range, intercept_unconf + slope_unconf * X_range, 'b--', label='Unconfounded Regression')
plt.legend()
plt.show()

# %%
