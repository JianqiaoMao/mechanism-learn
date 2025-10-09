#%%
import numpy as np
import matplotlib.pyplot as plt
import warnings
try:
    from IPython import get_ipython
    shell = get_ipython().__class__.__name__
    if shell == "ZMQInteractiveShell":
        # Jupyter notebook / JupyterLab
        from tqdm.notebook import tqdm
    else:
        # Other interactive shells
        from tqdm import tqdm
except (NameError, ImportError):
    # Ordinary Python shell
    from tqdm import tqdm
import pickle
import os

def log_gaussian_pdf(X, mu, Sigma, cov_type='full', cov_reg=1e-6):
    """
    Compute the log of the Gaussian probability density function.
    
    Parameters:
    X: shape=(N, d) data points
    mu: shape=(d,) mean vector
    Sigma: shape=(d, d) covariance matrix or shape=(d,) for diagonal covariance
    cov_type: 'full', 'diag', or 'spherical', default is 'full'
    cov_reg: regularization term for covariance matrix, default is 1e-6
    
    Return: 
    logp: shape=(N,) log-probability density function values
    """
    N, d = X.shape
    diff = X - mu

    if cov_type == 'full':
        if Sigma.shape == ():
            Sigma = np.array([[Sigma]])
        Sigma_reg = Sigma + cov_reg * np.eye(Sigma.shape[0])

        sign, logdet = np.linalg.slogdet(Sigma_reg)
        if sign <= 0:
            print("Warning: Sigma not positive definite.")
            return -np.inf * np.ones(N)

        inv_Sigma = np.linalg.inv(Sigma_reg)
        quadform = np.sum((diff @ inv_Sigma) * diff, axis=1)
        logp = -0.5 * (d * np.log(2.0 * np.pi) + logdet + quadform)

    elif cov_type == 'diag':
        Sigma_safe = np.maximum(Sigma, cov_reg)
        inv_Sigma = 1.0 / Sigma_safe
        quadform = np.sum((diff ** 2) * inv_Sigma, axis=1)
        logdet = np.sum(np.log(Sigma_safe))
        logp = -0.5 * (d * np.log(2.0 * np.pi) + logdet + quadform)

    elif cov_type == 'spherical':
        Sigma = np.asarray(Sigma)
        Sigma_safe = max(Sigma.item(), cov_reg)
        inv_Sigma = 1.0 / Sigma_safe
        quadform = np.sum(diff ** 2, axis=1) * inv_Sigma
        logdet = d * np.log(Sigma_safe)
        logp = -0.5 * (d * np.log(2.0 * np.pi) + logdet + quadform)
    
    else:
        raise ValueError("Parameter cov_type must be 'full', 'diag' or 'spherical'!")
          
    return  logp

def kmeans_plus_plus_init(X, weights, K, random_seed=None):
    """
    K-means++ initialization for GMM.
    
    Parameters:
    X: shape=(N, d) training data
    weights: shape=(N,) non-negative weights for each data point
    K: int, number of components
    random_seed: int, random seed for reproducibility, default is None
    
    Return: 
    centers: shape=(K, d) initial centers
    """
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    N, d = X.shape
    w = weights / np.sum(weights) 
    
    idx0 = np.random.choice(N, p=w)
    centers = [X[idx0].copy()]
    
    if K == 1:
        return np.array(centers)
    
    dist_sq = np.full(N, np.inf)
    
    for center_id in range(1, K):
        current_center = centers[-1] 
        diff = X - current_center
        dists = np.sum(diff*diff, axis=1)  # shape=(N,)
        dist_sq = np.minimum(dist_sq, dists)
        
        weighted_probs = dist_sq * w
        weighted_probs_sum = np.sum(weighted_probs)
        if weighted_probs_sum <= 1e-16:
            idx = np.random.choice(N)
        else:
            weighted_probs /= weighted_probs_sum
            idx = np.random.choice(N, p=weighted_probs)
        
        centers.append(X[idx].copy())
    
    return np.array(centers)

def weighted_gmm_em(X, weights, K,
                    cov_type='full',
                    cov_reg=1e-6,
                    min_variance_value=1e-6,
                    max_iter=1000,
                    tol=1e-7,
                    init_method='random',
                    user_assigned_mus=None,
                    random_seed=None,
                    show_progress_bar=True):

    if random_seed is not None:
        np.random.seed(random_seed)

    N, d = X.shape

    w_raw = np.asarray(weights, float)
    N_eff = float(np.sum(w_raw))
    if N_eff <= 0:
        raise ValueError("All weights are zero or negative.")
    w = w_raw / N_eff

    # —— Initialize pi, mus, Sigmas —— #
    pi = np.ones(K) / K
    if init_method == 'random':
        idx = np.random.choice(N, K, replace=False)
        mus = X[idx].copy()
    elif init_method == 'kmeans++':
        mus = kmeans_plus_plus_init(X, w, K, random_seed)
    elif init_method == 'user_assigned':
        mus = user_assigned_mus.copy()
    else:
        raise ValueError("init_method must be 'random', 'kmeans++' or 'user_assigned'!")

    # Initialize Sigmas
    overall_cov = np.cov(X.T, aweights=w)

    if cov_type == 'full':
        if d == 1:
            overall_cov = np.array([[float(overall_cov)]], dtype=float)  # (1,1)
        else:
            overall_cov = np.asarray(overall_cov, dtype=float)           # (d,d)
        Sigmas = np.repeat(overall_cov[None, :, :], K, axis=0)           # (K,d,d)

    elif cov_type == 'diag':
        if d == 1:
            overall_cov = np.array([float(overall_cov)], dtype=float)    # (1,)
        else:
            overall_cov = np.diag(np.asarray(overall_cov, dtype=float))  # (d,)
        Sigmas = np.repeat(overall_cov[None, :], K, axis=0)              # (K,d)

    else:  # spherical
        if d == 1:
            overall_cov = float(overall_cov)                             # scalar
        else:
            overall_cov = float(np.mean(np.diag(np.asarray(overall_cov, dtype=float))))
        Sigmas = np.full((K,), overall_cov, dtype=float)                 # (K,)

    logliks = []
    prev_loglik = -np.inf
    pbar = tqdm(range(max_iter), desc="EM iter", disable=not show_progress_bar)

    for it in pbar:
        # —— E step：evaluate log_pdf (N, K) —— #
        if cov_type == 'full':
            Sigma_reg = Sigmas + cov_reg * np.eye(d)[None, :, :]
            sign, logdet = np.linalg.slogdet(Sigma_reg)          # (K,)
            inv_S = np.linalg.inv(Sigma_reg)                     # (K, d, d)
            diffs = X[None, :, :] - mus[:, None, :]              # (K, N, d)
            difvinv = np.einsum('knd,kde->kne', diffs, inv_S)    # (K, N, d)
            quad = np.einsum('knd,knd->kn', difvinv, diffs)      # (K, N)
            const = d * np.log(2*np.pi)
            log_pdf = -0.5 * (const + logdet[:, None] + quad)    # (K, N)

        elif cov_type == 'diag':
            Sigma_safe = np.maximum(Sigmas, cov_reg)             # (K, d)
            inv_S = 1.0 / Sigma_safe
            logdet = np.sum(np.log(Sigma_safe), axis=1)          # (K,)
            diffs = X[None, :, :] - mus[:, None, :]              # (K, N, d)
            quad = np.einsum('knd,kd->kn', diffs**2, inv_S)      # (K, N)
            const = d * np.log(2*np.pi)
            log_pdf = -0.5 * (const + logdet[:, None] + quad)    # (K, N)

        else:  # spherical
            Sigma_safe = np.maximum(Sigmas, cov_reg)             # (K,)
            inv_S = 1.0 / Sigma_safe
            logdet = d * np.log(Sigma_safe)                      # (K,)
            diffs = X[None, :, :] - mus[:, None, :]              # (K, N, d)
            quad = np.sum(diffs**2, axis=2) * inv_S[:, None]     # (K, N)
            const = d * np.log(2*np.pi)
            log_pdf = -0.5 * (const + logdet[:, None] + quad)    # (K, N)

        log_pdf = log_pdf.T                                      # (N, K)

        pi_safe = np.clip(pi, 1e-12, None)
        log_prob = log_pdf + np.log(pi_safe)[None, :]            # (N, K)
        mx = np.max(log_prob, axis=1, keepdims=True)
        lse = mx + np.log(np.exp(log_prob - mx).sum(axis=1, keepdims=True))  # (N,1)
        loglik = float(np.sum(w[:, None] * lse))
        logliks.append(loglik)

        # responsibilities
        resp = np.exp(log_prob - mx)
        resp /= resp.sum(axis=1, keepdims=True)                  # (N, K)

        # —— M step：update pi, mus, Sigmas —— #
        Wk = np.sum(w[:, None] * resp, axis=0)                   # (K,)

        Wk = np.maximum(Wk, 1e-12)
        pi = Wk / np.sum(Wk)

        mus = (resp.T * w).dot(X) / Wk[:, None]                  # (K, d)

        if cov_type == 'full':
            diffs = X[None, :, :] - mus[:, None, :]              # (K, N, d)
            alpha = resp.T * w                                   # (K, N)
            difw = diffs * alpha[:, :, None]                     # (K, N, d)
            covs = np.einsum('kni,knj->kij', difw, diffs)        # (K, d, d)
            covs /= Wk[:, None, None]
            covs += cov_reg * np.eye(d)[None, :, :]

            eigvals, eigvecs = np.linalg.eigh(covs)              # (K, d), (K, d, d)
            eigvals = np.maximum(eigvals, min_variance_value)
            Sigmas = np.einsum('kij,kj,klj->kil', eigvecs, eigvals, eigvecs)  # (K, d, d)

        elif cov_type == 'diag':
            diffs = X[None, :, :] - mus[:, None, :]              # (K, N, d)
            alpha = resp.T * w                                   # (K, N)
            var_kd = np.sum(alpha[:, :, None] * (diffs**2), axis=1) / Wk[:, None]  # (K, d)
            Sigmas = np.maximum(var_kd + cov_reg, min_variance_value)             # (K, d)

        else:  # spherical
            diffs = X[None, :, :] - mus[:, None, :]              # (K, N, d)
            alpha = resp.T * w                                   # (K, N)
            var_k = np.sum(alpha[:, :, None] * (diffs**2), axis=(1, 2)) / (Wk * d)  # (K,)
            Sigmas = np.maximum(var_k + cov_reg, min_variance_value)               # (K,)

        # —— Check convergence —— #
        pbar.set_postfix({"Δloglik": f"{abs(loglik - prev_loglik):.2e}"})
        if abs(loglik - prev_loglik) < tol:
            break
        prev_loglik = loglik

    # —— AIC / BIC —— #
    if cov_type == 'full':
        cov_params = K * d * (d + 1) // 2
    elif cov_type == 'diag':
        cov_params = K * d
    else:
        cov_params = K
    k_params = (K - 1) + K * d + cov_params

    AIC = 2 * k_params - 2 * loglik
    BIC = np.log(max(N_eff, 1.0)) * k_params - 2 * loglik
    avg_loglik = loglik / max(N_eff, 1.0)

    return pi, mus, Sigmas, logliks, avg_loglik, AIC, BIC


def sample_from_gmm(pi, mus, Sigmas, num_samples, cov_type='full'):
    """
    Sample from a Gaussian Mixture Model (GMM).
    
    parameters:
    pi: shape=(K,) mixing coefficients
    mus: shape=(K, d) means of each component
    Sigmas: shape=(K, d, d) covariance matrices of each component
    num_samples: int, number of samples to generate
    cov_type: 'full', 'diag', or 'spherical', default is 'full'
    
    return: samples: shape=(num_samples, d) generated samples
    """
    
    K = len(pi)
    d = len(mus[0])
    samples_list = []

    comp_indices = np.random.choice(K, size=num_samples, p=pi)

    for k in range(K):
        count_k = np.sum(comp_indices == k)
        if count_k == 0:
            continue
        if cov_type == 'full':
            samples_k = np.random.multivariate_normal(mean=mus[k], cov=Sigmas[k], size=count_k)
        elif cov_type == 'diag':
            samples_k = np.random.multivariate_normal(mean=mus[k], cov=np.diag(Sigmas[k]), size=count_k)
        elif cov_type == 'spherical':
            samples_k = np.random.multivariate_normal(mean=mus[k], cov=np.eye(d)*Sigmas[k], size=count_k)
        else:
            raise ValueError("cov_type must be 'full', 'diag' or 'spherical'!")
        samples_list.append(samples_k)

    samples = np.vstack(samples_list)
    return samples

def load_CWGMMs(model_path):
    """
    Load a CWGMMs model from a file.
    Parameters:
    model_path: str, path to the model file
    Returns:
    cw_gmms: CWGMMs object with loaded parameters, scores and fit_flag
    """
    with open(model_path, 'rb') as f:
        model_info = pickle.load(f)
    cw_gmms = CWGMMs()
    cw_gmms.params = model_info['params']
    cw_gmms.scores = model_info['scores']
    cw_gmms.fit_flag = model_info['fit_flag']
    
    if not cw_gmms.fit_flag:
        warnings.warn("The loaded CWGMMs model is not fitted yet!")
    
    return cw_gmms

def compare_nested_params(param_dict1, param_dict2):
    """
    Compare two nested dictionaries of parameters.
    """

    if set(param_dict1.keys()) != set(param_dict2.keys()):
        return False

    for key in param_dict1:
        v1, v2 = param_dict1[key], param_dict2[key]


        if isinstance(v1, dict) and isinstance(v2, dict):
            if not compare_nested_params(v1, v2):
                return False
        
        elif isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
            if not np.array_equal(v1, v2):
                return False
        
        else:
            if v1 != v2:
                return False
    
    return True

class CWGMMs:
    
    """
    A class to store and sample from a stack of GMMs corresponding to different intervention values.
    
    Attributes:
    params: dict, stores GMM parameters for each model
    scores: dict, stores AIC, BIC and average log-likelihood scores for each model
    fit_flag: bool, indicates whether the model has been fitted
    
    Methods:
    score_update: update the scores for a GMM model
    write: write the GMM parameters to the params dictionary
    save: save the model parameters and scores to a file
    sample: sample from the GMMs
    evaluate_density: evaluate the density of the GMMs at given query points
    """

    def __init__(self):
        self.params = {}
        self.scores = {}
        self.fit_flag = False

    def score_update(self, model_name, AIC, BIC, avg_loglik_score):
        """
        Store the score for a GMM model.
        
        Parameters:
        model_name: str, name of the GMM model
        score: float, score value to store
        """
        self.scores[model_name] = {
            'AIC': AIC,
            'BIC': BIC,
            "avg_loglik_score": avg_loglik_score
        }
        
    def write(self, model_name, intv_value, pi, mus, Sigmas, cov_type):
        """
        Write the GMM parameters to the params dictionary.
        """
        self.params[model_name] = {
            'intv_value': intv_value,
            'pi': pi,
            'mus': mus,
            'Sigmas': Sigmas,
            'cov_type': cov_type
        }
        self.fit_flag = True
    
    def save(self, f_name, path = None):
        if path is None:
            path = os.getcwd()
        else:
            path = os.path.abspath(path)
        
        if not os.path.exists(path):
            os.makedirs(path)
        
        f_name = os.path.join(path, f_name + '.pkl')
        
        with open(f_name, 'wb') as f:
            # Save the parameters, scores and fit_flag
            model_info_to_save = {
                'params': self.params,
                'scores': self.scores,
                'fit_flag': self.fit_flag
            }
            pickle.dump(model_info_to_save, f)
        
        # Check if the file was saved successfully
        test_load_model = load_CWGMMs(f_name)
        consistency_check = compare_nested_params(self.params, test_load_model.params) and \
                            (self.scores == test_load_model.scores) and \
                            (self.fit_flag == test_load_model.fit_flag)
        if not consistency_check:
            # delete the file if it was not saved correctly
            os.remove(f_name)
            raise ValueError("Failed to save the model correctly.")
        else:
            print(f"Model saved successfully at {f_name}.")

    def sample(self, n_samples):
        if isinstance(n_samples, int):
            n_samples = [n_samples for _ in range(len(self.params))]
        elif len(n_samples) != len(self.params):
            raise ValueError("length of n_samples must be equal to the number of models!")
        new_samples = []
        intv_values = []

        for (model_name, params), n in zip(self.params.items(), n_samples):
            pi = params['pi']
            mus = params['mus']
            Sigmas = params['Sigmas']
            cov_type = params['cov_type']
            intv_value = params['intv_value']

            samples = sample_from_gmm(pi, mus, Sigmas, n, cov_type=cov_type)
            intv_value_n = [intv_value for _ in range(n)]
            new_samples.append(samples)
            intv_values.append(intv_value_n)
        
        new_samples = np.vstack(new_samples)
        intv_values = np.concatenate(intv_values, axis=0)
        intv_values = intv_values.reshape(-1, 1)
        return new_samples, intv_values
    
    def evaluate_density(self, x_query_batch):
        """
        Evaluate p(x_i | Y_j) for each x_i in x_query_batch and each GMM model.

        Parameters:
        x_query_batch: shape=(N, d) numpy array

        Returns:
        p_matrix: shape=(N, M) array where entry (i, j) = p(x_i | Y_j)
        """
        X = np.atleast_2d(x_query_batch)
        N, d = X.shape
        p_list = []

        for params in self.params.values():
            pi       = params['pi']       # (K,)
            mus      = params['mus']      # (K, d)
            Sigmas   = params['Sigmas']   # (K, d, d) / (K, d) / (K,)
            cov_type = params['cov_type']
            K = pi.shape[0]

            # compute log-pdf_{ik} in batch
            if cov_type == 'full':
                # (K, d, d) 
                sign, logdet = np.linalg.slogdet(Sigmas)      # (K,)
                invS = np.linalg.inv(Sigmas)                  # (K, d, d)

                # diffs: (K, N, d)
                diffs = X[None, :, :] - mus[:, None, :]
                # diffs @ invS -> (K, N, d)
                difvinv = np.einsum('knd,kde->kne', diffs, invS)
                # quadform: (K, N)
                quad = np.einsum('kne,kne->kn', difvinv, diffs)

                const = d * np.log(2*np.pi)
                # log_pdf_k: (N, K)
                log_pdf = (-0.5 * (const + logdet[:, None] + quad)).T

            elif cov_type == 'diag':
                # Sigmas: (K, d)
                invS = 1.0 / Sigmas                        # (K, d)
                logdet = np.sum(np.log(Sigmas), axis=1)    # (K,)

                diffs = X[None, :, :] - mus[:, None, :]        # (K, N, d)
                quad = np.einsum('knd,kd->kn', diffs**2, invS)  # (K, N)

                const = d * np.log(2*np.pi)
                log_pdf = (-0.5 * (const + logdet[:, None] + quad)).T

            else:  # spherical
                invS = 1.0 / Sigmas                       # (K,)
                logdet = d * np.log(Sigmas)               # (K,)

                diffs = X[None, :, :] - mus[:, None, :]       # (K, N, d)
                quad = np.sum(diffs**2, axis=2) * invS[:, None]# (K, N)

                const = d * np.log(2*np.pi)
                log_pdf = (-0.5 * (const + logdet[:, None] + quad)).T

            # log-sum-exp
            log_pi = np.log(pi + 1e-16)[None, :]             # (1, K)
            log_resp = log_pdf + log_pi                     # (N, K)
            m = np.max(log_resp, axis=1, keepdims=True)     # (N, 1)
            lse = m + np.log(np.sum(np.exp(log_resp - m), axis=1, keepdims=True) + 1e-16)
            p = np.exp(lse).flatten()                       # (N,)

            p_list.append(p)

        # return (N, M)
        return np.stack(p_list, axis=1)
        


# %%
