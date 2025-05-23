#%%
import causalBootstrapping as cb
from distEst_lib import MultivarContiDistributionEstimator
import numpy as np
import mechanism_learn.gmmSampler as gmms
import warnings
try:
    from IPython import get_ipython
    shell = get_ipython().__class__.__name__
    if shell == "ZMQInteractiveShell":
        # Jupyter notebook / JupyterLab
        from tqdm.notebook import tqdm
    else:
        # IPython Terminal, etc.
        from tqdm import tqdm
except (NameError, ImportError):
    # Standard Python interpreter
    from tqdm import tqdm
#%%

class mechanism_learning_process:
    def __init__(self, cause_data, mechanism_data, effect_data, intv_values, dist_map = None, est_method = "histogram", **est_kwargs):
        self.cause_data = cause_data.reshape(-1,1)
        self.mechanism_data = mechanism_data
        self.effect_data = effect_data
        self.intv_values = intv_values
        self.cause_var_name = "Y"
        self.mechanism_var_name = "Z"
        self.effect_var_name = "X"
        
        self.N = self.cause_data.shape[0]
        
        if dist_map is None:
            self.dist_map = self.dist_estimation(est_method = est_method, **est_kwargs)
        else:
            self.dist_map = dist_map
        self.causal_weights = self.frontdoor_causal_weights()
        
        self.deconf_X = np.zeros((0,self.effect_data.shape[1]))
        self.deconf_Y = np.zeros((0,self.cause_data.shape[1]))
        self.deconf_model = None
        
    def dist_estimation(self, est_method = "histogram", **est_kwargs):
        """
        This function computes the causal weights for the mechanism classifier.
        
        Parameters:
        dist_map: dict, Default: None
            A dictionary containing the distribution functions. The key is the variable name and the value is the distribution function. If None, fit intended distributions using simple histogram.
        est_method: str, Default: "histogram"
            The method to estimate the distribution. It can be "histogram", "kde" or "multinorm".
        **est_kwargs: dict
            The keyword arguments for the distribution estimator.
        
        Returns:
        dist_map: dict
            A dictionary containing the distribution functions. The key is the variable name and the value is the distribution function.
        """
            
        joint_yz_data = np.concatenate((self.cause_data, self.mechanism_data), axis = 1)
        if "n_bins" in est_kwargs:
            n_bins = est_kwargs["n_bins"]
        else:
            n_bins = [0 for i in range(joint_yz_data.shape[1])]
        yz_bins = n_bins.copy()
        y_bins = n_bins.copy()[:self.cause_data.shape[1]]
        dist_estimator_yz = MultivarContiDistributionEstimator(data_fit=joint_yz_data, n_bins = yz_bins)
        dist_estimator_y = MultivarContiDistributionEstimator(data_fit=self.cause_data, n_bins = y_bins)
        if est_method == "histogram":
            pdf_yz, _ = dist_estimator_yz.fit_histogram()
            pdf_y, _ = dist_estimator_y.fit_histogram()
        elif est_method == "kde":
            if "bandwidth" in est_kwargs:
                bandwidth = est_kwargs["bandwidth"]
            else:
                bandwidth = "scott"
            dist_estimator_yz.n_bins = [2 for i in range(joint_yz_data.shape[1])]
            dist_estimator_y.n_bins = [2 for i in range(self.cause_data.shape[1])]
            pdf_yz, _ = dist_estimator_yz.fit_kde(bandwidth)
            pdf_y, _ = dist_estimator_y.fit_kde(bandwidth)
        elif est_method == "multinorm":
            dist_estimator_yz.n_bins = [2 for i in range(joint_yz_data.shape[1])]
            dist_estimator_y.n_bins = [2 for i in range(self.cause_data.shape[1])]
            pdf_yz, _ = dist_estimator_yz.fit_multinorm()
            pdf_y, _ = dist_estimator_y.fit_multinorm()
        else:
            raise ValueError("Invalid estimation method. Choose 'histogram', 'kde' or 'multinorm'.")
            
        dist_map = {
            "Y,Z": lambda Y, Z: pdf_yz([Y,Z]),
            "Y',Z": lambda Y_prime, Z: pdf_yz([Y_prime,Z]),
            "Y": lambda Y: pdf_y(Y),
            "Y'": lambda Y_prime: pdf_y(Y_prime)
        }
        return dist_map
    
    def frontdoor_causal_weights(self):
        """
        This function computes the causal weights for the mechanism classifier.
        
        Parameters:
        None
        
        Returns:
        causal_weights: dict
            A dictionary containing the causal weights. The key is the variable name and the value is the causal weight.
        """
        
        data = {self.cause_var_name+"'": self.cause_data,
                self.mechanism_var_name: self.mechanism_data,
                self.effect_var_name: self.effect_data}
        
        causal_graph = self.cause_var_name + ";" + self.effect_var_name + ";" + self.mechanism_var_name + "; \n"
        causal_graph = causal_graph + self.cause_var_name + "->" + self.mechanism_var_name + "; \n"
        causal_graph = causal_graph + self.mechanism_var_name + "->" + self.effect_var_name + "; \n"
        causal_graph = causal_graph + self.cause_var_name + "<->" + self.effect_var_name + "; \n"
        
        weight_func_lam, _ = cb.general_cb_analysis(causal_graph = causal_graph, 
                                                    effect_var_name = self.effect_var_name, 
                                                    cause_var_name = self.cause_var_name, info_print= False)
                
        w_func = weight_func_lam(dist_map = self.dist_map, N = self.N, kernel = None)
        
        for i, intv_value in enumerate(self.intv_values):
            causal_weights_i = cb.weight_compute(weight_func = w_func,
                                                 data = data,
                                                 intv_var = {self.cause_var_name: [intv_value for j in range(self.N)]})
            causal_weights_i = causal_weights_i.reshape(-1,1)
            if i == 0:
                causal_weights = causal_weights_i
            else:
                # causal_weights shape: (N, num_intv_values)
                causal_weights = np.concatenate((causal_weights, causal_weights_i), axis = 1)
        
        return causal_weights
        
    def cwgmm_resample(self, comp_k, n_samples = None, 
                       max_iter = 1000, tol = 1e-4, init_method = "kmeans++", 
                       cov_type = "full", cov_reg = 1e-6, min_variance_value=1e-6, 
                       random_seed=None, verbose = 2, return_model = False, return_samples = False
                       ):
        """
        This function performs the CW-GMM resampling.
        
        Parameters:
        comp_k: int or list of int
            The number of components for the GMM. If a list, it should have the same length as intv_values.
        n_samples: int or list of int, Default: None
            The number of samples to be generated. If a list, it should have the same length as intv_values.
        max_iter: int or list of int, Default: 1000
            The maximum number of iterations for the GMM fitting. If a list, it should have the same length as intv_values.
        tol: float or list of float, Default: 1e-4
                The tolerance for the GMM fitting. If a list, it should have the same length as intv_values.
        init_method: str or list of str, Default: "kmeans++"
            The initialization method for the GMM fitting. If a list, it should have the same length as intv_values.
        cov_type: str or list of str, Default: "full"
            The covariance type for the GMM fitting. If a list, it should have the same length as intv_values.
        cov_reg: float or list of float, Default: 1e-6
            The covariance regularization for the GMM fitting. If a list, it should have the same length as intv_values.
        min_variance_value: float or list of float, Default: 1e-6
                The minimum variance value for the GMM fitting. If a list, it should have the same length as intv_values.
        random_seed: int, Default: None
            The random seed for the GMM fitting.
        verbose: int, Default: 2
            The verbosity level for the GMM fitting. 0: no progress bar, 1: show GMM progress bar, 2: show both GMM and intv progress bar.
        return_model: bool, Default: False
            Whether to return the fitted GMM model.
        return_samples: bool, Default: False
            Whether to return the generated samples.
        
        Returns:
        deconf_X: np.ndarray
            The generated samples of X so as the deconfounded X.
        deconf_Y: np.ndarray
            The corresponding interventional values so as the deconfounded Y.
        cwgmms: object
            The fitted GMM model.
        """
        
        if isinstance(comp_k, int):
            comp_k = [comp_k for i in range(len(self.intv_values))]
        if isinstance(max_iter, float) or isinstance(max_iter, int):
            max_iter = [max_iter for i in range(len(self.intv_values))]
        if isinstance(tol, float) or isinstance(tol, int):
            tol = [tol for i in range(len(self.intv_values))]
        if isinstance(init_method, str):
            init_method = [init_method for i in range(len(self.intv_values))]
        if isinstance(cov_type, str):
            cov_type = [cov_type for i in range(len(self.intv_values))]
        if isinstance(cov_reg, float) or isinstance(cov_reg, int):
            cov_reg = [cov_reg for i in range(len(self.intv_values))]
        if isinstance(min_variance_value, float) or isinstance(min_variance_value, int):
            min_variance_value = [min_variance_value for i in range(len(self.intv_values))]
        if return_samples and n_samples is None:
            raise ValueError("n_samples should be specified when return_samples is True.")
        if isinstance(n_samples, int):
            n_samples = [n_samples for i in range(len(self.intv_values))]
        if len(comp_k) != len(self.intv_values):
            raise ValueError("comp_k should be a list of the same length as intv_values.")
        if verbose == 2:
            show_gmm_progress_bar = True
            show_intv_progress_bar = True
        elif verbose == 1:
            show_gmm_progress_bar = False
            show_intv_progress_bar = True
        elif verbose == 0:
            show_gmm_progress_bar = False
            show_intv_progress_bar = False
        else:
            raise ValueError("verbose should be 0 or 1.")
        cwgmms = gmms.CWGMMs()
        deconf_X = np.zeros((0,self.effect_data.shape[1]))
        deconf_Y = np.zeros((0,self.cause_data.shape[1]))
        pbar = tqdm(enumerate(self.intv_values), total = len(self.intv_values), desc = "CW-GMM Resampling", disable = not show_intv_progress_bar)
        for i, intv_value in pbar:
            causal_weights_i = self.causal_weights[:,i]*self.N
            pi_est_intv, mus_est_intv, Sigmas_est_intv, _, _ = gmms.weighted_gmm_em(
                                                               self.effect_data, causal_weights_i, K=comp_k[i], 
                                                               cov_type=cov_type[i], max_iter=max_iter[i], tol=tol[i], 
                                                               init_method=init_method[i],
                                                               cov_reg=cov_reg[i], min_variance_value=min_variance_value[i],
                                                               random_seed=random_seed, show_progress_bar=show_gmm_progress_bar)   
            if return_model:
                cwgmms.write(i, intv_value, pi_est_intv, mus_est_intv, Sigmas_est_intv, cov_type[i])    
            deconf_samples_intv_i = gmms.sample_from_gmm(pi_est_intv, mus_est_intv, Sigmas_est_intv,
                                                        n_samples[i], cov_type=cov_type[i])
            deconf_X = np.concatenate((deconf_X, deconf_samples_intv_i), axis = 0)
            deconf_Y = np.concatenate((deconf_Y, np.array([intv_value for j in range(n_samples[i])]).reshape(-1,1)), axis = 0)
        
        sample_idx = np.arange(deconf_X.shape[0])
        np.random.shuffle(sample_idx)
        self.deconf_X = deconf_X[sample_idx]
        self.deconf_Y = deconf_Y[sample_idx]
        
        if return_model and return_samples:
            return self.deconf_X, self.deconf_Y, cwgmms
        if return_samples:
            return self.deconf_X, self.deconf_Y    
        if return_model:
            return cwgmms
            
    def cb_resample(self, n_samples, cb_mode = "fast", return_samples = False, random_seed=None, verbose = 1):
        """
        This function performs the causal bootstrapping resampling.
        
        Parameters:
        n_samples: int or list of int
            The number of samples to be generated. If a list, it should have the same length as intv_values.
        cb_mode: str, Default: "fast"
            The mode for the causal bootstrapping. It can be "fast" or "full".
        return_samples: bool, Default: False
            Whether to return the generated samples.
        random_seed: int, Default: None
            The random seed for the causal bootstrapping.
        verbose: int, Default: 1
            The verbosity level for the causal bootstrapping. 0: no progress bar, 1: show intv progress bar.
                    
        Returns:
        deconf_X: np.ndarray
            The generated samples of X so as the deconfounded X.
        deconf_Y: np.ndarray
            The corresponding interventional values so as the deconfounded Y.
        """
        
        if isinstance(n_samples, int):
            n_samples = [n_samples for i in range(len(self.intv_values))]
        if verbose == 1:
            show_intv_progress_bar = True
        elif verbose == 0:
            show_intv_progress_bar = False
        else:
            raise ValueError("verbose should be 0 or 1.")
        
        cb_data = {}
        pbar = tqdm(enumerate(self.intv_values), total = len(self.intv_values), desc = "CB Resampling", disable = not show_intv_progress_bar)
        for i, intv_value in pbar:
            cause_data = {self.cause_var_name: self.cause_data}
            mediator_data = {self.mechanism_var_name: self.mechanism_data}
            effect_data = {self.effect_var_name: self.effect_data}
            cb_data_i = cb.frontdoor_simu(cause_data = cause_data,
                                          mediator_data = mediator_data,
                                          effect_data = effect_data,
                                          dist_map = self.dist_map,
                                          mode = cb_mode,
                                          n_sample = n_samples[i],
                                          intv_value = [intv_value for j in range(self.N)],
                                          random_state=random_seed)
            for key in cb_data_i:
                if i == 0:
                    cb_data[key] = cb_data_i[key]
                else:
                    cb_data[key] = np.vstack((cb_data[key], cb_data_i[key]))
        deconf_X = cb_data[self.effect_var_name]
        deconf_Y = cb_data["intv_"+self.cause_var_name].reshape(-1,1)

        sample_idx = np.arange(deconf_X.shape[0])
        np.random.shuffle(sample_idx)
        self.deconf_X = deconf_X[sample_idx]
        self.deconf_Y = deconf_Y[sample_idx]
        
        if return_samples:
            return self.deconf_X, self.deconf_Y
    
    def deconf_model_fit(self, ml_model):
        """
        This function fits the deconfounded model.
        
        Parameters:
        ml_model: object
            A classifier or regressor object. It should have the fit method.
 
        Returns:
        ml_model: object
            The fitted machine learning model.
        """
        
        self.deconf_model = ml_model.fit(self.deconf_X, self.deconf_Y.ravel())
        return self.deconf_model
    
        
    
    