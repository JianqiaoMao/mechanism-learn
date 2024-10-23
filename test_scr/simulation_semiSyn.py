#%%
import sys
sys.path.append("E:/Happiness_source/PhD/UoB/projects/Mechanism Learning/code")
%matplotlib qt5

#%%
import pandas as pd
import numpy as np
from distEst_lib import MultivarContiDistributionEstimator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import mechanism_learning as ml
semiSyn_data_dir = "E:/Happiness_source/PhD/UoB/projects/Mechanism Learning/code/dataset/simu_data/semi_synthetic_data/frontdoor_data/"
#%%

X_train_conf = pd.read_csv(semiSyn_data_dir + "X_train_conf.csv")
Y_train_conf = pd.read_csv(semiSyn_data_dir + "Y_train_conf.csv")
Z_train_conf = pd.read_csv(semiSyn_data_dir + "Z_train_conf.csv")
X_train_conf = np.array(X_train_conf)
Y_train_conf = np.array(Y_train_conf).reshape(-1,1)
Z_train_conf = np.array(Z_train_conf).reshape(-1,1)

X_test_unconf = pd.read_csv(semiSyn_data_dir + "X_test_unconf.csv")
Y_test_unconf = pd.read_csv(semiSyn_data_dir + "Y_test_unconf.csv")
X_test_unconf = np.array(X_test_unconf)
Y_test_unconf = np.array(Y_test_unconf).reshape(-1,1)

X_test_conf = pd.read_csv(semiSyn_data_dir + "X_test_conf.csv")
Y_test_conf = pd.read_csv(semiSyn_data_dir + "Y_test_conf.csv")
X_test_conf = np.array(X_test_conf)
Y_test_conf = np.array(Y_test_conf).reshape(-1,1)
#%%
# train the deconfounded svc
clf_deconf, deconf_data = ml.mechanism_classifier(cause_data = {"Y": Y_train_conf}, 
                                                  mediator_data = {"Z": Z_train_conf},
                                                  effect_data = {"X": X_train_conf}, 
                                                  n_bins = [0,0],
                                                  ml_model = KNeighborsClassifier(n_neighbors = 5), 
                                                  rebalance = False, 
                                                  n_samples = None, 
                                                  cb_mode = "fast",
                                                  output_data = True)

# train the confounded svc
clf_conf = KNeighborsClassifier(n_neighbors = 5)
clf_conf = clf_conf.fit(X_train_conf, Y_train_conf.reshape(-1))

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

#%%