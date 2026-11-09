import numpy as np
from scipy.stats import norm
import time

_t0 = time.time()

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're running out of time, return a simple score quickly
    elapsed = time.time() - _t0
    if elapsed > 320.0:
        # Fallback: just use -mu (exploit lowest mean) with minimal cost penalty
        mu = np.asarray(mu, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        cost_safe = np.maximum(cost, 1e-9)
        score = -mu / cost_safe
        return np.where(np.isfinite(score), score, 0.0)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Expected improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB exploration term: encourages visiting high-uncertainty regions
    beta = 1.5
    ucb = -mu + beta * sigma_safe  # higher = more promising (lower mu, higher sigma)
    
    # Adaptive cost weighting: stronger penalty when f_best is close to 0 (late stage)
    f_best_scale = max(abs(f_best), 1e-6)
    alpha = 1.0 / (1.0 + f_best_scale)
    
    cost_weight = cost_safe ** (-alpha)
    
    # Blend: primarily cost-weighted EI, with a smaller UCB component
    # The UCB term helps escape local optima and explore uncertain regions
    # Weight UCB more when f_best is high (early stage) and less when low (late stage)
    ucb_weight = 0.1 / (1.0 + f_best_scale)
    
    score = ei * cost_weight + ucb_weight * ucb * cost_weight
    
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    return score
