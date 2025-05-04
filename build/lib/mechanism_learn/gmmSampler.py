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

# @profile
def weighted_gmm_em(X, weights, K, cov_type = 'full', cov_reg = 1e-6, min_variance_value=1e-6, max_iter=1000, tol=1e-7, init_method='random', user_assigned_mus = None, random_seed=None):
    """
    Using EM algorithm to fit a weighted GMM model.
    
    Parameters:
    X: shape=(N, d) training data
    weights: shape=(N,) non-negative weights for each data point
    K: int, number of components
    cov_type: 'full', 'diag', or 'spherical', default is 'full'
    cov_reg: regularization term for covariance matrix, default is 1e-6
    min_variance_value: minimum variance value for covariance matrix, default is 1e-6
    max_iter: int, maximum number of iterations, default is 1000
    tol: float, tolerance for convergence, default is 1e-7
    init_method: 'random', 'kmeans++', or 'user_assigned', default is 'random'
    user_assigned_mus: shape=(K, d), user assigned initial means, only used when init_method='user_assigned'
    random_seed: int, random seed for reproducibility, default is None
    
    Return: pi, mus, Sigmas, logliks, avg_loglik_score
    pi: shape=(K,) mixing coefficients
    mus: shape=(K, d) means of each component
    Sigmas: shape=(K, d, d) covariance matrices of each component
    logliks: list of log-likelihood values at each iteration
    avg_loglik_score: average log-likelihood score
    
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    N, d = X.shape
    w = np.asarray(weights, dtype=float)
    if len(w) != N:
        raise ValueError("The length of weights must be equal to the number of data points!")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative!")
    total_weight = np.sum(w)
    
    # Initialization: pi, mus, Sigmas
    pi = np.ones(K) / K
    if init_method == 'random':
        rand_idxs = np.random.choice(N, K, replace=False)
        mus = X[rand_idxs].copy()
    elif init_method == 'kmeans++':
        mus = kmeans_plus_plus_init(X, w, K, random_seed=random_seed)
    elif init_method == 'user_assigned':
        mus = user_assigned_mus
    else:
        raise ValueError("init_method 参数错误!")
        
    overall_cov = np.cov(X.T, aweights=w)
    
    if d == 1:
        if cov_type == 'diag':
            overall_cov = np.array([overall_cov])  # shape=(1,)
        elif cov_type == 'spherical':
            overall_cov = float(overall_cov)
        elif cov_type == 'full':
            overall_cov = np.array([[overall_cov]])  # shape=(1,1)
    else:
        if cov_type == 'diag':
            overall_cov = np.diag(overall_cov)
        elif cov_type == 'spherical':
            overall_cov = np.mean(np.diag(overall_cov))
        elif cov_type != 'full':
            raise ValueError("cov_type 参数错误!")
    Sigmas = np.array([overall_cov.copy() for _ in range(K)])

    logliks = []
    prev_loglik = -np.inf
    pbar = tqdm(range(max_iter), desc="E-M optimization", unit="itr", leave=False)
    for it in pbar:
        # E-step
        # Compute responsibility score
        log_resp = np.zeros((N, K))
        log_pdf_k = np.empty((N, K))
        for k in range(K):
            log_pdf_k[:,k] = log_gaussian_pdf(X, mus[k], Sigmas[k], cov_type=cov_type, cov_reg=cov_reg)
            log_resp[:, k] = np.log(pi[k] + 1e-16) + log_pdf_k[:,k]

        # Softmax for numerical stability
        max_log_resp = np.max(log_resp, axis=1, keepdims=True)  # shape=(N,1)
        log_resp -= max_log_resp
        resp = np.exp(log_resp)
        denom = np.sum(resp, axis=1, keepdims=True) + 1e-16
        resp /= denom

        # M-step
        Wk = np.sum(w[:, None] * resp, axis=0)  # shape=(K,)
        pi = Wk / np.sum(Wk)
        mus_new = np.zeros((K, d))
        if cov_type == 'full':
            Sigmas_new = np.zeros((K, d, d))
        elif cov_type == 'diag':
            Sigmas_new = np.zeros((K, d))
        elif cov_type == 'spherical':
            Sigmas_new = np.zeros(K)
        min_variance_value = max(min_variance_value, np.var(X)/1e3)

        for k in range(K):
            if Wk[k] < 1e-10:
                rand_idx = np.random.choice(N)
                mus_new[k] = X[rand_idx].copy()
                Sigmas_new[k] = overall_cov if cov_type == 'full' else np.maximum(overall_cov, min_variance_value)
                continue

            mus_new[k] = np.sum(w[:,None] * resp[:,k][:,None] * X, axis=0) / Wk[k]
            diff = X - mus_new[k]
            if cov_type == 'full':
                cov_k = np.zeros((d,d))
                alpha = w * resp[:,k]
                diff_weighted = diff * alpha[:, None]
                cov_k = diff_weighted.T @ diff
                cov_k /= Wk[k]
                cov_k += cov_reg * np.eye(d)

                eigvals, eigvecs = np.linalg.eigh(cov_k)
                eigvals = np.maximum(eigvals, min_variance_value)
                cov_k = eigvecs @ np.diag(eigvals) @ eigvecs.T
                Sigmas_new[k] = cov_k

            elif cov_type == 'diag':
                cov_k = np.sum((w * resp[:,k])[:, None] * diff**2, axis=0) / Wk[k]
                cov_k += cov_reg
                Sigmas_new[k] = np.maximum(cov_k, min_variance_value)

            elif cov_type == 'spherical':
                cov_k = np.sum(w * resp[:,k] * np.sum(diff**2, axis=1)) / (d * Wk[k])
                cov_k += cov_reg
                Sigmas_new[k] = max(cov_k, min_variance_value)
        
        mus = mus_new
        Sigmas = Sigmas_new

        # Compute weighted log-likelihood
        log_pdf_new = np.empty((N, K))
        for k in range(K):
            log_pdf_new[:,k] = log_gaussian_pdf(X, mus[k], Sigmas[k], cov_type=cov_type, cov_reg=cov_reg)

        log_pdf_weighted = log_pdf_new + np.log(pi + 1e-16)  # shape=(N, K)
        max_log = np.max(log_pdf_weighted, axis=1, keepdims=True)
        logsumexp = max_log + np.log(np.sum(np.exp(log_pdf_weighted - max_log), axis=1) + 1e-16)
        loglik = np.sum(w * logsumexp)
        avg_loglik_score = loglik / total_weight
        logliks.append(loglik)
        pbar.set_postfix({"loglik dif.": "{:.2e}".format(np.abs(loglik - prev_loglik))})
        # Check convergence
        if np.abs(loglik - prev_loglik) < tol:
            break
        prev_loglik = loglik
        if it == max_iter-1:
            warnings.warn("Warning: EM reached max_iter without convergence!")
    pbar.close()

    return pi, mus, Sigmas, logliks, avg_loglik_score

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

class CWGMMs:
    
    """
    A class to store and sample from multiple GMMs.
    """

    def __init__(self):
        self.params = {}
        
    def write(self, model_name, intv_value, pi, mus, Sigmas, cov_type):
        self.params[model_name] = {
            'intv_value': intv_value,
            'pi': pi,
            'mus': mus,
            'Sigmas': Sigmas,
            'cov_type': cov_type
        }
    
    def save(self, f_name, path = None):
        if path is None:
            path = os.getcwd()
        else:
            path = os.path.abspath(path)
        
        if not os.path.exists(path):
            os.makedirs(path)
        
        f_name = os.path.join(path, f_name + '.pkl')
        
        with open(f_name, 'wb') as f:
            pickle.dump(self.params, f)
    
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
        


# %%
