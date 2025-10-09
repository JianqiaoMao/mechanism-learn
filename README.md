# mechanism-learn [![DOI](https://zenodo.org/badge/877382338.svg)](https://doi.org/10.5281/zenodo.13984553)
Mechanism-learn is a user-friendly Python library which uses front-door causal bootstrapping to deconfound observational data such that any appropriate machine learning (ML) model is forced to learn predictive relationships between effects and their causes (reverse causal inference), despite the potential presence of multiple unknown and unmeasured confounding. The library is compatible with most existing machine learning deployments such as scikit-learn, Keras.

## Mechanism Learning

One of the major limitations of applying ML methods in critical, high-stakes applications such as decision-making in medicine, is their "causal blindness", that is, ML models are, by design, pattern-recognition algorithms which learn potentially spurious, non-causal associations between the feature and target variables. 

<div align="center">
  <img src="./figures/mechanism_learning_new.png" alt="figure1" width="1100">
</div>

<p align="left"><strong>Figure 1.</strong> Mechanism learning (b) is a novel, simple and widely applicable solution to the problem of reverse causal inference in the presence of multiple unknown confounding, using arbitrary supervised ML algorithms to predict nonlinear effect-cause relationships from potentially high-dimensional effects. The causal scenario is represented by the ubiquitous front-door causal graph (a). There are multiple, unmeasured/unknown confounding paths between $Y$ and $X$ (bi-directed, dashed arrow). The classic causal inference direction is the causal path from $Y$ to $X$ via $Z$ (blue half arrow); reverse causal inference infers causes $Y$ from effects $X$ (red half arrow).</p>


To address the challenge of "causal-blindness" in ML models, mechanism learning provides a solution. a simple method which uses front-door causal bootstrapping to deconfound observational data such that any appropriate ML model is forced to learn predictive relationships between effects and their causes (reverse causal inference), despite the potential presence of multiple unknown and unmeasured confounding. This novel method is widely applicable, the only requirement is the existence of a mechanism variable mediating the cause (prediction target) and effect (feature data), which is independent of the (unmeasured) confounding variables. 

For example, in radiology, there are unique morphological features caused by different diseases which lead to specific patterns in the digital images, while many unmeasured and/or unknown confounders exist such as the imaging device protocol and patient demographics which simultaneously influence both the diagnostic category and the images. The following figure shows a real-world application of intracranial hemorrhage detection using CT scans, where mechanism learning can well fit. So we can apply mechanism learning to force a ML model to learn the causal relationships between the diagnositic category (cause variable) and the corresponding CT scans (effect variable), and thus make causal hemorrhage diagnosis for an input CT scan from a potential patient.

<div align="center">
  <img src="./figures/IH_mechanism_learning_representation.png" alt="figure1" width="600">
</div>

<p align="left"><strong>Figure 2.</strong> Front-door structural causal model for the real-world ICH dataset [1], for the purposes of mechanism learning. The cause variable $Y$ represents diagnostic category; mechanism variable $Z$ represents hemorrhage region label, and the effect variable $X$ are the digital CT scans.</p>

## Citing

Please use one of the following to cite the code of this repository.

```
@article{mao2024mechanism,
  title={Mechanism learning: Reverse causal inference in the presence of multiple unknown confounding through front-door causal bootstrapping},
  author={Mao, Jianqiao and Little, Max A},
  journal={arXiv preprint arXiv:2410.20057},
  year={2024}
}
```

## Installation and getting started

We currently offer seamless installation with `pip`. 

Simply:
```
pip install mechanism-learn
```

Alternatively, download the current distribution of the package, and run:
```
pip install .
```
in the root directory of the decompressed package.

To import the package:
```python
from mechanism_learn import pipeline as mlpipe
```

## Demo. for classification task

Please refer to other [examples](https://github.com/JianqiaoMao/mechanism-learn/tree/main/test_scr) for detialed instructions and usage demonstrations.

1. Import mechanismlearn lib and other libs for demo. and define some functions for convenience.
```python
from mechanism_learn import pipeline as mlpipe
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def linear_decision_boundary(clf, xlim):
    weight = clf.coef_[0]
    bias = clf.intercept_[0]
    k = -weight[0] / weight[1]
    b = -bias / weight[1]
    x = np.linspace(xlim[0], xlim[1], 100)
    decison_boundary = k * x + b
    return decison_boundary

def plot_boundary(ax, X, Y, db, xlim, ylim, plot_title):
    handles = []
    scatter = ax.scatter(x= X[:,0], y = X[:,1], c = Y, s = 10, alpha = 0.4, cmap='coolwarm')
    handles_scatter, labels_scatter = scatter.legend_elements(prop="colors")
    handles += handles_scatter
    ax.set_xlim(xlim[0],xlim[1])
    ax.set_ylim(ylim[0],ylim[1])
    x_ = np.linspace(xlim[0], xlim[1], 100)
    
    true_b = ax.plot([0, 0], [-6, 6], 'k', linewidth=4, label="True boundary")
    confounder = ax.plot([-6,6], [0,0], 'orange', linewidth = 4, label = "Confounder boundary")
    clf_b = ax.plot(x_, db, 'r', linewidth = 4, label= 'Classifier Decision')
    handles += [true_b[0], confounder[0], clf_b[0]]
    ax.set_title(plot_title)
    ax.set_xlabel(r"$X_1$")
    ax.set_ylabel(r"$X_2$")
    return ax, handles

```

2. Read synthetic classification datasets
```python
syn_data_dir = r"../test_data/synthetic_data/"
testcase_dir = r"classification/"
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
```

Here, we have a confounded training set which mimics some observational dataset confounded in front-door criteria. Variables $Y$, $X$ and $Z$ represent the cause, effect and mechanism variables, respectively. Also, there are two test sets, where one is a confounded test set with the same confounding as the confounded training set, and another one is a non-conofunded test set without any confounding between the cause and effect variables.

3.  Train a deconfounded SVM classifier using mechanism learning
```python
ml_gmm_pipeline = mlpipe.mechanism_learning_process(cause_data = Y_train_conf,
                                                    mechanism_data = Z_train_conf, 
                                                    effect_data = X_train_conf, 
                                                    intv_values = np.unique(Y_train_conf), 
                                                    dist_map = None, 
                                                    est_method = "histogram", 
                                                    n_bins = [0, 10]
                                                    )
deconf_X_gmm, deconf_Y_gmm = ml_gmm_pipeline.cwgmm_resample(comp_k = 6,
                                                            n_samples = 5000,
                                                            max_iter = 500, 
                                                            tol = 1e-7, 
                                                            init_method = "kmeans++", 
                                                            cov_type = "full", 
                                                            cov_reg = 1e-6, 
                                                            min_variance_value=1e-6, 
                                                            random_seed=None, 
                                                            return_model = False, 
                                                            return_samples = True)

deconf_gmm_clf = ml_gmm_pipeline.deconf_model_fit(ml_model = svm.SVC(kernel = 'linear', C=5))
```

4. Train a deconfounded SVM classifier using CB-based deconfounding method
```python
ml_cb_pipeline = mlpipe.mechanism_learning_process(cause_data = Y_train_conf,
                                                   mechanism_data = Z_train_conf, 
                                                   effect_data = X_train_conf, 
                                                   intv_values = np.unique(Y_train_conf), 
                                                   dist_map = None, 
                                                   est_method = "histogram", 
                                                   n_bins = [0, 10]
                                                   )
deconf_X_cb, deconf_Y_cb = ml_cb_pipeline.cb_resample(n_samples = 5000,
                                                      cb_mode = "fast",
                                                      return_samples = True)

deconf_cb_clf = ml_cb_pipeline.deconf_model_fit(ml_model = svm.SVC(kernel = 'linear', C=5))
```

In this toy example, a linear SVM is used. Meanwhile, the default histogram method is used to estimate the required distributions for mechanism learning (specifically, $\hat p\left(z|y\right)$ ). Although the confounded training data is input, mechanism learning will deconfound the given confounded dataset and train the SVM which can make causal predictions. Refer to the reference for more information.

5. Train a confounded SVM classifier
```python
conf_clf = svm.SVC(kernel = 'linear', C=5)
conf_clf = conf_clf.fit(X_train_conf, Y_train_conf.reshape(-1))
```

For comparison, this code segmentation trains a naive SVM which is a confounded model because of the confounded training data.

5. Compare their decision boundaries

```python
decision_boundary_deconf_gmm = linear_decision_boundary(clf = deconf_gmm_clf, xlim = [-4,4])
decision_boundary_deconf_cb = linear_decision_boundary(clf = deconf_cb_clf, xlim = [-4,4])
decision_boundary_conf = linear_decision_boundary(clf = conf_clf, xlim = [-4,4])
```

```python
fig, axes = plt.subplots(3,1, figsize = (6,16))
axes[0], _ = plot_boundary(axes[0], deconf_X_gmm, deconf_Y_gmm, 
                        decision_boundary_deconf_gmm, [-6,6], [-6,6], 'Decision boundary of the SVM using mechanism learning \n and the corresponding deconfounded dataset')
axes[1], _ = plot_boundary(axes[1], deconf_X_cb, deconf_Y_cb, 
                        decision_boundary_deconf_cb, [-6,6], [-6,6], 'Decision boundary of the SVM using CB deconfounding \n and the corresponding deconfounded dataset')
axes[2], handles = plot_boundary(axes[2], X_train_conf, Y_train_conf, 
                        decision_boundary_conf, [-6,6], [-6,6], 'Decision boundary of the confounded SVM \n with the confounded dataset')
axes[0].tick_params(labelbottom=False)
axes[0].set_xlabel('')  
axes[1].tick_params(labelbottom=False)
axes[1].set_xlabel('')  
labels = [r'$Y=1$ (class 1)', r'$Y=2$ (class 2)',
          'True class boundary', 'Confounder boundary', 'SVM decision boundary']
legend =fig.legend(handles=handles,
                   labels=labels,
                   loc='lower center',
                   bbox_to_anchor=(0.5, 0.08),
                   ncol=5,
                   markerscale=2)
for handle in legend.legendHandles:
    handle.set_alpha(1.0)
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.show()
```

By running these cells, you may expect to have the output similar to below:

<div align="center">
  <img src="./figures/syn_clas_results.png" alt="figure3" width="700">
</div>

<p align="left"><strong>Figure 3.</strong> Comparing classifiers trained using (a) mechanism learning, (b) CB-based deconfounding and (c) classical supervised learning, on the synthetic classification dataset. With confounded data (c), class 1 samples tend to concentrate in the bottom left region, with class 2 samples in the top right. The SVM classifier's decision boundary mixes the true with the confounding boundary. By contrast, after applying CW-GMM resampling (a) and front-door CB (b) to resample the confounded dataset, the confounding factor has been nullified, such that the SVM decision boundary is close to the true class boundary. However, CW-GMM resampling has better sample variability compared to the front-door CB. </p>

The confounded SVM, trained using classical supervised learning, is severely affected by the confounder, whose decision boundary conflates the true class with the implied confounder boundaries. By contrast, the decision boundaries of the mechanism learning-based deconfounded and CB-based deconfounded SVMs closely match the true class boundary, which nullifies the influence of the confounder. Therefore, both of the deconfounded models capture the desired causal relationship between features and prediction target.

6. Compare their performance in evaluation metrics on confounded and non-confounded test sets

```python
print("Test on the non-confounded test set:")

y_pred_gmm_deconf_unconf = deconf_gmm_clf.predict(X_test_unconf)
print("Report of deconfounded model using mechanism learning:")
print(classification_report(Y_test_unconf, y_pred_gmm_deconf_unconf, digits=4))
print("-"*20)
y_pred_cb_deconf_unconf = deconf_cb_clf.predict(X_test_unconf)
print("Report of deconfounded model using CB-based method:")
print(classification_report(Y_test_unconf, y_pred_cb_deconf_unconf, digits=4))
print("-"*20)
y_pred_conf_unconf = conf_clf.predict(X_test_unconf)
print("Report of confonded model:")
print(classification_report(Y_test_unconf, y_pred_conf_unconf, digits=4))


print("*"*30)
print("Test on the confounded test set:")

y_pred_gmm_deconf_conf = deconf_gmm_clf.predict(X_test_conf)
print("Report of deconfounded model using mechanism learning:")
print(classification_report(Y_test_conf, y_pred_gmm_deconf_conf, digits=4))
print("-"*20)
y_pred_cb_deconf_conf = deconf_cb_clf.predict(X_test_conf)
print("Report of deconfounded model using CB-based method:")
print(classification_report(Y_test_conf, y_pred_cb_deconf_conf, digits=4))
print("-"*20)
y_pred_conf_conf = conf_clf.predict(X_test_conf)
print("Report of confonded model:")
print(classification_report(Y_test_conf, y_pred_conf_conf, digits=4))
```

The expected output should be similar to:

```output
Test on the non-confounded test set:
Report of deconfounded model using mechanism learning:
              precision    recall  f1-score   support

           1     0.7687    0.8536    0.8089       362
           2     0.9114    0.8542    0.8819       638

    accuracy                         0.8540      1000
   macro avg     0.8400    0.8539    0.8454      1000
weighted avg     0.8597    0.8540    0.8555      1000

--------------------
Report of deconfounded model using CB-based method:
              precision    recall  f1-score   support

           1     0.7681    0.8508    0.8073       362
           2     0.9098    0.8542    0.8812       638

    accuracy                         0.8530      1000
   macro avg     0.8390    0.8525    0.8443      1000
weighted avg     0.8585    0.8530    0.8544      1000

--------------------
Report of confonded model:
              precision    recall  f1-score   support

           1     0.6513    0.8564    0.7399       362
           2     0.9008    0.7398    0.8124       638

    accuracy                         0.7820      1000
   macro avg     0.7760    0.7981    0.7761      1000
weighted avg     0.8104    0.7820    0.7861      1000

******************************
Test on the confounded test set:
Report of deconfounded model using mechanism learning:
              precision    recall  f1-score   support

           1     0.9195    0.8497    0.8832       632
           2     0.7716    0.8723    0.8189       368

    accuracy                         0.8580      1000
   macro avg     0.8456    0.8610    0.8511      1000
weighted avg     0.8651    0.8580    0.8595      1000

--------------------
Report of deconfounded model using CB-based method:
              precision    recall  f1-score   support

           1     0.9194    0.8481    0.8823       632
           2     0.7698    0.8723    0.8178       368

    accuracy                         0.8570      1000
   macro avg     0.8446    0.8602    0.8501      1000
weighted avg     0.8643    0.8570    0.8586      1000

--------------------
Report of confonded model:
              precision    recall  f1-score   support

           1     0.8868    0.8797    0.8832       632
           2     0.7962    0.8071    0.8016       368

    accuracy                         0.8530      1000
   macro avg     0.8415    0.8434    0.8424      1000
weighted avg     0.8535    0.8530    0.8532      1000
```

In terms of performance evaluation metrics, the mechanism learning-based deconfounded model and CB-based deconfounded model exceed the SVM trained using classical supervised learning on the non-confounded dataset by 7% accuracy, retaining stable predictive accuracy across both confounded and non-confounded datasets. On the contrary, although the classical supervised SVM performs about as well as the mechanism learning-based SVM in the confounded dataset, its performance declines significantly on the non-confounded dataset. This is because classical supervised learning is biased by the influence of confounding, reporting misleading accuracy on the original (confounded) test set. However, due to the simplicity and low dimensionality of the synthetic data, the performance of CB-based and mechanism learning-based deconfounded models remains largely indistinguishable.

### Reference

**[1]** Hssayeni, M. (2020). [Computed Tomography Images for Intracranial Hemorrhage Detection and Segmentation (version 1.3.1)](https://doi.org/10.13026/4nae-zg36). PhysioNet.
