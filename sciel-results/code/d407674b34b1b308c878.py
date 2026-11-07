"""Cost-aware acquisition for Bayesian optimization of Powell 4D."""
import time
import numpy as np
from scipy.stats import norm

_DEADLINE = time.time() + 325.0


def acquisition(mu, sigma, f_best, cost):
    if time.time() > _DEADLINE:
        return -mu / (cost + 1e-9)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma = np.maximum(sigma, 1e-8)
    
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Progress: f_best starts high, goes to 0
    # explore_weight high when f_best is large (early), low when f_best is small (late)
    # Use a smooth transition centered around f_best ~ 1-5
    explore_weight = 1.0 / (1.0 + np.exp((f_best - 2.0) / 1.0))
    exploit_weight = 1.0 - explore_weight
    
    # Normalize EI
    ei_max = np.max(ei) + 1e-9
    ei_norm = ei / ei_max
    
    # Exploration: favor high uncertainty
    sigma_max = np.max(sigma) + 1e-9
    expl_score = sigma / sigma_max
    
    # Log-cost penalty
    log_cost = np.log(cost + 1e-9)
    
    # Combine
    score = exploit_weight * ei_norm + explore_weight * expl_score * 0.5 - 0.3 * log_cost
    
    score = np.where(np.isfinite(score), score, -1e6)
    return score
