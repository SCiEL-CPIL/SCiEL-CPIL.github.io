import numpy as np
from scipy.stats import norm
import time

_START_TIME = None

def _get_elapsed_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def acquisition(mu, sigma, f_best, cost):
    if _get_elapsed_time() > 290:
        mu = np.asarray(mu)
        cost = np.asarray(cost)
        cost_safe = np.maximum(cost, 1e-9)
        scores = -mu / cost_safe
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    z = (f_best - mu) / sigma_safe
    cdf_val = norm.cdf(z)
    pdf_val = norm.pdf(z)
    
    exploit = (f_best - mu) * cdf_val
    explore = sigma_safe * pdf_val
    
    # Adaptive alpha
    positive_gap = np.maximum(f_best - mu, 0.0)
    eps = 1e-6
    uncertainty_ratio = sigma_safe / (positive_gap + eps)
    uncertainty_ratio = np.clip(uncertainty_ratio, 0.0, 10.0)
    
    # Alpha ranges from 0.1 (very exploitative) to 0.9 (very exploratory)
    alpha = 0.1 + 0.8 * np.tanh(uncertainty_ratio / 2.0)
    
    ei_adapt = exploit + alpha * explore
    score = ei_adapt / cost_safe
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
