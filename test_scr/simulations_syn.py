#%% Import libraries and specify directories

import pandas as pd
import numpy as np
from distEst_lib import MultivarContiDistributionEstimator
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import mechanismlearn as ml
import matplotlib.collections
syn_data_dir = "E:/Happiness_source/PhD/UoB/projects/Mechanism Learning/code/dataset/simu_data/synthetic_data/"

#%% Simu classification: frontdoor_discY_contZ_contX_discU
testcase_dir = "frontdoor_discY_contZ_contX_discU/"
X_train_conf = pd.read_csv(syn_data_dir + testcase_dir + "X_train_conf.csv")
Y_train_conf = pd.read_csv(syn_data_dir + testcase_dir + "Y_train_conf.csv")
Z_train_conf = pd.read_csv(syn_data_dir + testcase_dir + "Z_train_conf.csv")
X_train_conf = np.array(X_train_conf)
Y_train_conf = np.array(Y_train_conf).reshape(-1,1)
Z_train_conf = np.array(Z_train_conf)

X_test_unconf = pd.read_csv(syn_data_dir + testcase_dir + "X_test_unconf.csv")
Y_test_unconf = pd.read_csv(syn_data_dir + testcase_dir + "Y_test_unconf.csv")
X_test_unconf = np.array(X_test_unconf)
Y_test_unconf = np.array(Y_test_unconf).reshape(-1,1)

X_test_conf = pd.read_csv(syn_data_dir + testcase_dir + "X_test_conf.csv")
Y_test_conf = pd.read_csv(syn_data_dir + testcase_dir + "Y_test_conf.csv")
X_test_conf = np.array(X_test_conf)
Y_test_conf = np.array(Y_test_conf).reshape(-1,1)

# train the deconfounded svc
clf_deconf, deconf_data = ml.mechanism_classifier(cause_data = {"Y": Y_train_conf}, 
                                                  mediator_data = {"Z": Z_train_conf},
                                                  effect_data = {"X": X_train_conf}, 
                                                  n_bins = [0,20],
                                                  ml_model = svm.SVC(kernel = 'linear', C=5), 
                                                  rebalance = False, 
                                                  n_samples = None, 
                                                  cb_mode = "fast",
                                                  output_data = True)

# train the confounded svc
clf_conf = svm.SVC(kernel = 'linear', C=5)
clf_conf = clf_conf.fit(X_train_conf, Y_train_conf.reshape(-1))

#%% Plot the decision boundaries and the data points
# Model decision boundaries
weight = clf_deconf.coef_[0]
bias = clf_deconf.intercept_[0]
k = -weight[0] / weight[1]
b = -bias / weight[1]
x_ = np.linspace(-4, 4, 100)
decison_boundary_deconf = k * x_ + b

weight = clf_conf.coef_[0]
bias = clf_conf.intercept_[0]
k = -weight[0] / weight[1]
b = -bias / weight[1]
x_ = np.linspace(-4, 4, 100)
decison_boundary_conf = k * x_ + b
plt.rcParams['font.family'] = 'Calibri'
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["axes.labelsize"] = 18
# plot on deconfounded test set
fig,ax=plt.subplots(figsize = (6.1, 5.9))
scatter = ax.scatter(x= deconf_data[0][:,0], y = deconf_data[0][:,1], c = deconf_data[1], s = 10, cmap='coolwarm', alpha = 0.5,
                     label = "Deconfounded test set")
handles_scatter, labels_scatter = scatter.legend_elements(prop="colors")

plt.xlim(-7,7)
plt.ylim(-8,8)
x_ = np.linspace(-4, 4, 100)

true_b = plt.plot([0, 0], [-10, 10], linewidth=4, color = "black", label="True boundary")
confounder = plt.plot([-10,10], [0,0], linewidth = 4, color = "orange", label = "Confounder boundary")
clf_b_deconf = plt.plot(x_, decison_boundary_deconf,  linewidth = 5, color = "red", label= 'De-confounded SVM decision boundary')
plt.subplots_adjust(bottom=0.2)
ax.set_xlabel(r"$X_1$")
ax.set_yticks([])
plt.tight_layout()
# plt.savefig("syn_simu_classification_deconf.png", dpi=600 ,bbox_inches='tight')
plt.show()

figLegend = plt.figure(figsize=(12, 2))
ax = figLegend.add_subplot(111)
ax.axis('off')  
legend = ax.legend(handles=handles_scatter + true_b + confounder + clf_b_deconf,
                   labels=[r'$Y=1$ (class 1)',
                           r'$Y=2$ (class 2)',
                           'True class boundary',
                           'Confounder boundary',
                           'SVM decision boundary'],
                   loc='center',
                   fontsize=16,
                   frameon=False,
                   ncol=5,
                   markerscale=2)
for handle in legend.legendHandles:
    handle.set_alpha(1.0)
figLegend.savefig('legend.png', bbox_inches='tight')
plt.close(figLegend) 

# Plot on confounded test set
fig,ax=plt.subplots(figsize = (6.7, 6))
scatter = ax.scatter(x= X_train_conf[:,0], y = X_train_conf[:,1], c = Y_train_conf, s = 10, cmap='coolwarm', alpha = 0.5,
                     label = "Confounded test set")

handles_scatter, labels_scatter = scatter.legend_elements(prop="colors")

plt.xlim(-7,7)
plt.ylim(-8,8)
x_ = np.linspace(-4, 4, 100)

true_b = plt.plot([0, 0], [-10, 10], linewidth=4, color = "black", label="True boundary")
confounder = plt.plot([-10,10], [0,0], linewidth = 4, color = "orange", label = "Confounder boundary")
clf_b_conf = plt.plot(x_, decison_boundary_conf, linewidth = 5, color = "red", label= 'Confounded SVM decision boundary')
ax.set_xlabel(r"$X_1$")
ax.set_ylabel(r"$X_2$")
plt.tight_layout()
# plt.savefig("syn_simu_classification_conf.png", dpi=600 ,bbox_inches='tight')
plt.show()

#%% Metrics comparison
print("Test on the unconfounded test set:")
y_pred_deconf_unconf = clf_deconf.predict(X_test_unconf)
print("Report of de-confonded model:")
print(classification_report(Y_test_unconf, y_pred_deconf_unconf, digits = 4))

print("-"*20)
y_pred_conf_unconf = clf_conf.predict(X_test_unconf)
print("Report of confonded model:")
print(classification_report(Y_test_unconf, y_pred_conf_unconf, digits = 4))

print("*"*30)
print("Test on the confounded test set:")
y_pred_deconf_conf = clf_deconf.predict(X_test_conf)
print("Report of de-confonded model:")
print(classification_report(Y_test_conf, y_pred_deconf_conf, digits = 4))

print("-"*20)
y_pred_conf_conf = clf_conf.predict(X_test_conf)
print("Report of confonded model:")
print(classification_report(Y_test_conf, y_pred_conf_conf, digits = 4))


#%% Simu regression: frontdoor_contY_contZ_contX_contU

testcase_dir = "frontdoor_contY_contZ_contX_contU/"
X_train_conf = pd.read_csv(syn_data_dir + testcase_dir + "X_train_conf.csv")
Y_train_conf = pd.read_csv(syn_data_dir + testcase_dir + "Y_train_conf.csv")
U_train_conf = pd.read_csv(syn_data_dir + testcase_dir + "U_train_conf.csv")
Z_train_conf = pd.read_csv(syn_data_dir + testcase_dir + "Z_train_conf.csv")
X_train_conf = np.array(X_train_conf).reshape(-1,1)
Y_train_conf = np.array(Y_train_conf).reshape(-1,1)
U_train_conf = np.array(U_train_conf).reshape(-1,1)
Z_train_conf = np.array(Z_train_conf).reshape(-1,1)

X_test_conf = pd.read_csv(syn_data_dir + testcase_dir + "X_test_conf.csv")
Y_test_conf = pd.read_csv(syn_data_dir + testcase_dir + "Y_test_conf.csv")
X_test_conf = np.array(X_test_conf).reshape(-1,1)
Y_test_conf = np.array(Y_test_conf).reshape(-1,1)

X_test_unconf = pd.read_csv(syn_data_dir + testcase_dir + "X_test_unconf.csv")
Y_test_unconf = pd.read_csv(syn_data_dir + testcase_dir + "Y_test_unconf.csv")
X_test_unconf = np.array(X_test_unconf).reshape(-1,1)
Y_test_unconf = np.array(Y_test_unconf).reshape(-1,1)

cause_data = {"Y": Y_train_conf}
mediator_data = {"Z": Z_train_conf}
effect_data = {"X": X_train_conf}

reg_lr_deconf = LinearRegression()

N = X_train_conf.shape[0]
intv_intval_num = 50
Y_interv_values = np.linspace(np.mean(Y_train_conf) - 1.2*(np.max(Y_train_conf)- np.min(Y_train_conf)), 
                              np.mean(Y_train_conf) + 1.2*(np.max(Y_train_conf)- np.min(Y_train_conf)), 
                              intv_intval_num+1)
n_samples = [int(N/intv_intval_num)]*intv_intval_num

joint_yz_data = np.concatenate((Y_train_conf, Z_train_conf), axis = 1)
N = X_train_conf.shape[0]
dist_estimator_yz = MultivarContiDistributionEstimator(data_fit=joint_yz_data)
pdf_yz, pyz = dist_estimator_yz.fit_multinorm()
dist_estimator_y = MultivarContiDistributionEstimator(data_fit=Y_train_conf)
pdf_y, py = dist_estimator_y.fit_multinorm()

dist_map = {"Y,Z": lambda Y, Z: pdf_yz([Y,Z]),
            "Y',Z": lambda Y_prime, Z: pdf_yz([Y_prime,Z]),
            "Y": lambda Y: pdf_y(Y),
            "Y'": lambda Y_prime: pdf_y(Y_prime)}

# Train the deconfounded model
reg_lr_deconf, deconf_data = ml.mechanism_regressor(cause_data = cause_data,
                                                    mediator_data = mediator_data,
                                                    effect_data = effect_data,
                                                    ml_model = reg_lr_deconf,
                                                    intv_value = Y_interv_values, 
                                                    n_samples = n_samples, 
                                                    dist_map = dist_map, 
                                                    cb_mode = "fast",
                                                    output_data = True)

# Train the confounded model
reg_lr_conf = LinearRegression()
reg_lr_conf = reg_lr_conf.fit(X_train_conf, Y_train_conf.reshape(-1))

# Train the unconfounded model
reg_lr_unconf = LinearRegression()
reg_lr_unconf = reg_lr_unconf.fit(X_test_unconf, Y_test_unconf.reshape(-1))

#%% Visualize the regression lines and the data points

# Plot the regression lines
x_grid = np.linspace(np.min([np.min(X_train_conf), np.min(X_test_unconf)]),np.max([np.max(X_train_conf), np.max(X_test_unconf)]), num=1000)
regLine_conf = reg_lr_conf.predict(x_grid.reshape(-1,1))
regLine_deconf = reg_lr_deconf.predict(x_grid.reshape(-1,1))
regLine_unconf = reg_lr_unconf.predict(x_grid.reshape(-1,1))
plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18
plt.rcParams["axes.labelsize"] = 20
plt.figure(figsize = (8,5))
plt.scatter(X_test_conf, Y_test_conf, s = 10, c = 'royalblue', alpha = 0.3, label = "Confounded data")
plt.scatter(X_test_unconf, Y_test_unconf, s = 10, c = 'mediumseagreen', alpha = 0.3, label = "Non-confounded data")
plt.plot(x_grid, regLine_conf, 'red', label = "Confounded regression line", linewidth = 3)
plt.plot(x_grid, regLine_unconf, 'black', label='Non-confounded regression line', linewidth = 3)
plt.xlabel(r"$X$")
plt.ylabel(r"$Y$")
plt.ylim(-12, 20)
plt.xlim(-25, 32)
plt.tight_layout()
legend = plt.legend(fontsize = 16, loc = "upper left", markerscale = 2, framealpha = 0.3)
for handle in legend.legendHandles:
    if isinstance(handle, matplotlib.collections.PathCollection):
        handle.set_alpha(1.0)
plt.savefig("syn_simu_reg_conf.png", dpi=600 ,bbox_inches='tight')
plt.show()

plt.figure(figsize = (7.3,4.9))
plt.scatter(deconf_data[0], deconf_data[1], s = 10, c = 'orangered', alpha = 0.3, label = "Deconfounded data")
plt.scatter(X_test_unconf, Y_test_unconf, s = 10, c = 'mediumseagreen', alpha = 0.3, label = "Non-confounded data")
# plt.scatter(X_test_conf, Y_test_conf, s = 10, c = 'royalblue', alpha = 0.3, label = "Confounded data")
plt.plot(x_grid, regLine_deconf, 'red', label = "Deconfounded regression line", linewidth = 3)
# plt.plot(x_grid, regLine_conf, 'darkblue', label = "Confounded regression line", linewidth = 2)
plt.plot(x_grid, regLine_unconf, 'black', label='Non-confounded regression line', linewidth = 3)
plt.xlabel(r"$X$")
plt.yticks([])
plt.ylim(-12, 20)
plt.xlim(-25, 32)
# plt.ylabel(r"$Y$")
# plt.xlim(-25, 32)
# plt.title("De-/Un-/Confounded regression lines with deconfounded and confounded data")
legend = plt.legend(fontsize = 16, loc = "upper left", markerscale = 2, framealpha = 0.3)
for handle in legend.legendHandles:
    if isinstance(handle, matplotlib.collections.PathCollection):
        handle.set_alpha(1.0)
plt.tight_layout()
plt.savefig("syn_simu_reg_deconf.png", dpi=600 ,bbox_inches='tight')
plt.show()
#%% Metrics comparison

# Test on the unconfounded test set
y_pred_deconf_unconf = reg_lr_deconf.predict(X_test_unconf)
y_pred_conf_unconf = reg_lr_conf.predict(X_test_unconf)

mse_deconf_unconf = mean_squared_error(Y_test_unconf, y_pred_deconf_unconf)
r2_deconf_unconf = r2_score(Y_test_unconf, y_pred_deconf_unconf)
mae_deconf_unconf = mean_absolute_error(Y_test_unconf, y_pred_deconf_unconf)

mse_conf_unconf = mean_squared_error(Y_test_unconf, y_pred_conf_unconf)
r2_conf_unconf = r2_score(Y_test_unconf, y_pred_conf_unconf)
mae_conf_unconf = mean_absolute_error(Y_test_unconf, y_pred_conf_unconf)

print("Test on the unconfounded test set:")
print("Deconfounded model:")
print(f"RMSE: {mse_deconf_unconf**0.5}")
print(f"R2: {r2_deconf_unconf}")
print(f"MAE: {mae_deconf_unconf}")

print("-"*20)
print("Confounded model:")
print(f"RMSE: {mse_conf_unconf**0.5}")
print(f"R2: {r2_conf_unconf}")
print(f"MAE: {mae_conf_unconf}")

# Test on the confounded test set
y_pred_deconf_conf = reg_lr_deconf.predict(X_test_conf)
y_pred_conf_conf = reg_lr_conf.predict(X_test_conf)

mse_deconf_conf = mean_squared_error(Y_test_conf, y_pred_deconf_conf)
r2_deconf_conf = r2_score(Y_test_conf, y_pred_deconf_conf)
mae_deconf_conf = mean_absolute_error(Y_test_conf, y_pred_deconf_conf)

mse_conf_conf = mean_squared_error(Y_test_conf, y_pred_conf_conf)
r2_conf_conf = r2_score(Y_test_conf, y_pred_conf_conf)
mae_conf_conf = mean_absolute_error(Y_test_conf, y_pred_conf_conf)

print("*"*30)
print("Test on the confounded test set:")
print("Deconfounded model:")
print(f"RMSE: {mse_deconf_conf**0.5}")
print(f"R2: {r2_deconf_conf}")
print(f"MAE: {mae_deconf_conf}")

print("-"*20)
print("Confounded model:")
print(f"RMSE: {mse_conf_conf**0.5}")
print(f"R2: {r2_conf_conf}")
print(f"MAE: {mae_conf_conf}")

# %%
