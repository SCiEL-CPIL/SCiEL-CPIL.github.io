import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle zero sigma and cost to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Expected Improvement (EI) for minimization
    # EI = max(0, f_best - mu) - sigma * phi(z) - (f_best - mu) * phi(z) / z where z = (f_best - mu) / sigma
    # More numerically stable version:
    imp = np.maximum(0.0, f_best - mu)
    z = imp / sigma_safe
    ei = imp * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei_per_cost = ei / cost_safe
    
    # Lower Confidence Bound (LCB) for exploitation
    # LCB = mu - k * sigma (lower is better for minimization)
    # Score: (k*sigma - mu) / cost
    k_lcb = 1.0
    lcb_score = (k_lcb * sigma - mu) / cost_safe
    
    # Cost-aware UCB (Upper Confidence Bound)
    # For minimization, we want low values, so UCB is not directly applicable
    # Instead, use a cost-adjusted exploration term
    # UCB-like: (f_best - mu + k_ucb * sigma) / cost
    k_ucb = 0.5
    ucb_score = (f_best - mu + k_ucb * sigma) / cost_safe
    
    # Normalize each score
    def normalize_scores(s):
        s_min = np.min(s)
        s_max = np.max(s)
        s_range = s_max - s_min
        if s_range > 1e-12:
            return (s - s_min) / s_range
        return np.zeros_like(s)
    
    norm_ei = normalize_scores(ei_per_cost)
    norm_lcb = normalize_scores(lcb_score)
    norm_ucb = normalize_scores(ucb_score)
    
    # Blend EI with LCB and UCB
    # EI provides expected improvement, LCB provides exploitation, UCB provides exploration
    w_ei = 0.5
    w_lcb = 0.3
    w_ucb = 0.2
    
    scores = w_ei * norm_ei + w_lcb * norm_lcb + w_ucb * norm_ucb
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    if time.time() - start_time > 325:
        return np.zeros_like(mu)
    
    return scores
