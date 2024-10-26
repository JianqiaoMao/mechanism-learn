#%%
import causalBootstrapping as cb
from distEst_lib import MultivarContiDistributionEstimator
import numpy as np
#%%
def mechanism_classifier(cause_data, 
                         mechanism_data, 
                         effect_data, 
                         ml_model, 
                         dist_map = None, 
                         n_bins = [20, 20],
                         rebalance = False, 
                         n_samples = None, 
                         cb_mode = "fast",
                         output_data = False):
    """
    This function trains a classification model to predict the effect variable given the cause variable and the mediator variable.
    
    Parameters:
    cause_data: dict
        A dictionary containing the cause variable data. The key is the variable name and the value is a n-d data array.
    mechanism_data: dict
        A dictionary containing the mechanism variable data. The key is the variable name and the value is a n-d data array.
    effect_data: dict   
        A dictionary containing the effect variable data. The key is the variable name and the value is a n-d data array.
    ml_model: object
        A classifier object. It should have the fit method.
    dist_map: dict, Default: None
        A dictionary containing the distribution functions. The key is the variable name and the value is the distribution function. If None, fit intended distributions using simple histogram.
    n_bins: list, Default: [20, 20]
        A list containing the number of bins for each variable. The first element is the number of bins for the effect variable and the second element is the number of bins for the mediator variable. If dist_map is not None, this parameter will be ignored.
    rebalance: bool, Default: False
        If True, the data will be rebalanced by the cause variable.
    n_samples: int, Default: None
        The number of samples to be generated for each value of the cause variable. If None, the number of samples will be the same for each value of the cause variable.
    cb_mode: str, Default: "fast"
        The mode of causal bootstrapping. It can be "fast" or "robust".
    output_data: bool, Default: False
        If True, the function will return the the deconfounded model, deconfounded data. If False, the function will only return the deconfounded model.
    
    Returns:
    ml_model: object
        The deconfounded machine learning model.
    (deconf_X, deconf_Y): tuple of arrays
        The deconfounded data. This will be returned only if output_data is True.
    """
    
    cause_var_name = list(cause_data.keys())[0]
    mechanism_var_name = list(mechanism_data.keys())[0]   
    effect_var_name = list(effect_data.keys())[0]

    Y = cause_data[cause_var_name]   
    X = effect_data[effect_var_name] 
    Z = mechanism_data[mechanism_var_name]

    if dist_map is None:
        
        cause_data = {"Y": Y}
        mechanism_data = {"Z": Z}
        effect_data = {"X": X}
        
        cause_var_name = list(cause_data.keys())[0]
        mechanism_var_name = list(mechanism_data.keys())[0]   
        effect_var_name = list(effect_data.keys())[0]
        
        joint_yz_data = np.concatenate((Y, Z), axis = 1)
        
        dist_estimator_yz = MultivarContiDistributionEstimator(data_fit=joint_yz_data, n_bins = n_bins)
        pdf_yz, _ = dist_estimator_yz.fit_histogram()
        dist_estimator_y = MultivarContiDistributionEstimator(data_fit=Y, n_bins = [n_bins[0]])
        pdf_y, _ = dist_estimator_y.fit_histogram()
        
        dist_map = {
            "Y,Z": lambda Y, Z: pdf_yz([Y,Z]),
            "Y',Z": lambda Y_prime, Z: pdf_yz([Y_prime,Z]),
            "Y": lambda Y: pdf_y(Y),
            "Y'": lambda Y_prime: pdf_y(Y_prime)
        }
    
    if rebalance:
        cause_unique = np.unique(Y)
        if n_samples is None:
            N = Y.shape[0]
            cause_unique_n = len(cause_unique)
            n_samples = [int(N/cause_unique_n)]*cause_unique_n
        cb_data = {}
        for i, interv_value in enumerate(cause_unique):
            cb_data_simu = cb.frontdoor_simu(cause_data = cause_data,
                                             mediator_data = mechanism_data,
                                             effect_data = effect_data,
                                             dist_map = dist_map,
                                             mode = cb_mode,
                                             n_samples = n_samples[i],
                                             interv_value = interv_value)
            for key in cb_data_simu:
                if i == 0:
                    cb_data[key] = cb_data_simu[key]
                else:
                    cb_data[key] = np.vstack((cb_data[key], cb_data_simu[key]))
    else:
        cb_data = cb.frontdoor_simple(cause_data = cause_data,
                                      mediator_data = mechanism_data,
                                      effect_data = effect_data,
                                      dist_map = dist_map,
                                      mode = cb_mode)
    deconf_X = cb_data[effect_var_name]
    deconf_Y = cb_data["intv_"+cause_var_name].ravel()
    ml_model = ml_model.fit(deconf_X, deconf_Y)
    
    if output_data:
        return ml_model, (deconf_X, deconf_Y)
    else:
        return ml_model

def mechanism_regressor(cause_data, 
                        mechanism_data, 
                        effect_data, 
                        ml_model, 
                        intv_value = None,
                        intv_intval_num = 50,
                        n_samples = None, 
                        dist_map = None, 
                        cb_mode = "fast",
                        output_data = False):
    """
    This function trains a regression model to predict the effect variable given the cause variable and the mediator variable.
    
    Parameters:
    cause_data: dict
        A dictionary containing the cause variable data. The key is the variable name and the value is a n-d data array.
    mechanism_data: dict
        A dictionary containing the mechanism variable data. The key is the variable name and the value is a n-d data array.
    effect_data: dict
        A dictionary containing the effect variable data. The key is the variable name and the value is a n-d data array.
    ml_model: object
        A regressor object. It should have the fit method.
    intv_value: list, Default: None
        A list containing the values of the cause variable to be used as the intervention values. If None, a set of even intervention values will be generated.
    intv_intval_num: int, Default: 50
        The number of even intervals to be used to generate the intervention values. This parameter will be used only if intv_value is None. 
    n_samples: int, Default: None
        The number of samples to be generated for each value of intervention. If None, the number of samples will be the N/intv_intval_num.
    dist_map: dict, Default: None
        A dictionary containing the distribution functions. The key is the variable name and the value is the distribution function. If None, fit intended distributions using kernel density estimation.
    cb_mode: str, Default: "fast"
        The mode of causal bootstrapping. It can be "fast" or "robust". 
    output_data: bool, Default: False
        If True, the function will return the the deconfounded model, deconfounded data. If False, the function will only return the deconfounded model.
    
    Returns:
    ml_model: object
        The deconfounded machine learning model.
    (deconf_X, deconf_Y): tuple of arrays
        The deconfounded data. This will be returned only if output_data is True.
    """
    
    cause_var_name = list(cause_data.keys())[0]
    mechanism_var_name = list(mechanism_data.keys())[0]   
    effect_var_name = list(effect_data.keys())[0]

    Y = cause_data[cause_var_name]   
    X = effect_data[effect_var_name] 
    Z = mechanism_data[mechanism_var_name]

    N = Y.shape[0]
    
    if dist_map is None:
        
        cause_data = {"Y": Y}
        mechanism_data = {"Z": Z}
        effect_data = {"X": X}
        
        cause_var_name = list(cause_data.keys())[0]
        mechanism_var_name = list(mechanism_data.keys())[0]   
        effect_var_name = list(effect_data.keys())[0]
        
        joint_yz_data = np.concatenate((Y, Z), axis = 1)
        
        dist_estimator_yz = MultivarContiDistributionEstimator(data_fit=joint_yz_data, n_bins = [50, 50])
        pdf_yz, _ = dist_estimator_yz.fit_kde()
        dist_estimator_y = MultivarContiDistributionEstimator(data_fit=Y, n_bins = [50])
        pdf_y, _ = dist_estimator_y.fit_kde()
        
        dist_map = {
            "Y,Z": lambda Y, Z: pdf_yz([Y,Z]),
            "Y',Z": lambda Y_prime, Z: pdf_yz([Y_prime,Z]),
            "Y": lambda Y: pdf_y(Y),
            "Y'": lambda Y_prime: pdf_y(Y_prime)
        }
    
    if intv_value is None:
        
        intv_value = np.linspace(np.min(Y) - np.abs(np.min(Y)), 
                                 np.max(Y) + np.abs(np.max(Y)), 
                                 intv_intval_num+1)
        
    if n_samples is None:
        n_samples = [int(N/intv_intval_num)]*intv_intval_num
    
    cb_data = {}
    for i, interv_value in enumerate(intv_value[:-1]):
        cb_data_intv = cb.frontdoor_simu(cause_data = cause_data, 
                                         effect_data = effect_data, 
                                         mediator_data = mechanism_data, 
                                         dist_map = dist_map, 
                                         intv_value = [interv_value for j in range(N)], 
                                         n_sample = n_samples[i], 
                                         mode = cb_mode)
        for key in cb_data_intv:
            if i == 0:
                cb_data[key] = cb_data_intv[key]
            else:
                cb_data[key] = np.vstack((cb_data[key], cb_data_intv[key]))
    
    deconf_X = cb_data[effect_var_name]
    deconf_Y = cb_data["intv_"+cause_var_name].ravel()
    ml_model = ml_model.fit(deconf_X, deconf_Y)
    
    if output_data:
        return ml_model, (deconf_X, deconf_Y)
    else:
        return ml_model
    
    