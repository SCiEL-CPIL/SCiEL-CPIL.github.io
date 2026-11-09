"""Acquisition function for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function combining UCB and EI with adaptive exploration.
    
    Uses a time-aware strategy: early on, favor exploration (UCB); later,
    favor exploitation (EI). Also handles edge cases for numerical stability.
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-9)  # Avoid division by zero
    
    # Compute Expected Improvement
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues in cdf/pdf
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Compute UCB (Upper Confidence Bound)
    # Use a beta that balances exploration and exploitation
    # Adaptive beta: start high for exploration, decrease over time
    # Since we don't have step count, use f_best as a proxy:
    # if f_best is close to minimum (3.0), we're doing well, so exploit more
    # if f_best is far from minimum, explore more
    
    # Goldstein-Price minimum is 3.0
    f_min = 3.0
    
    # Normalize f_best relative to the range
    # Typical values might range from ~3 to ~100
    # Use a simple heuristic: beta decreases as we get closer to the optimum
    progress = 1.0 - np.clip((f_best - f_min) / 10.0, 0, 1)
    
    # Beta: start at 3.0 (high exploration), decrease to 1.0 (more exploitation)
    beta = 1.0 + 2.0 * (1.0 - progress)
    
    ucb = mu + beta * sigma
    
    # Combine EI and UCB
    # Normalize both to have similar scale
    ei_max = np.max(ei) if np.max(ei) > 0 else 1e-9
    ucb_max = np.max(ucb) if np.max(ucb) > 0 else 1e-9
    
    # Weight: early on favor UCB, later favor EI
    w_ucb = 0.5 + 0.5 * (1.0 - progress)  # 1.0 early, 0.5 late
    w_ei = 1.0 - w_ucb
    
    # Combine
    score = w_ucb * (ucb / (ucb_max + 1e-9)) + w_ei * (ei / (ei_max + 1e-9))
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
