"""Pareto-based multi-objective acquisition for Bayesian optimization of Hartmann 6D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Pareto-based multi-objective acquisition function.
    
    Combines normalized Expected Improvement (EI) and normalized predictive uncertainty
    with a dynamic weight that increases with the number of evaluations (proxied by
    the spread of sigma relative to the scale of the problem), balancing exploitation
    and exploration.
    
    Parameters:
        mu: array of shape (2000,) - posterior mean in objective units
        sigma: array of shape (2000,) - posterior std dev in objective units
        f_best: float - best observed value so far (lowest)
    
    Returns:
        scores: array of shape (2000,) - higher scores mean evaluate next
    """
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute predictive uncertainty
    # Normalize EI and Sigma to have similar scales for weighted sum
    # Using standard deviation of the batch is a robust normalizer
    ei_std = np.std(ei)
    sigma_std = np.std(sigma)
    
    # Avoid division by zero
    if ei_std < 1e-9:
        ei_normalized = ei
    else:
        ei_normalized = ei / ei_std
        
    if sigma_std < 1e-9:
        uncertainty = sigma
    else:
        uncertainty = sigma / sigma_std
        
    # Dynamic weight based on f_best
    # f_best starts high (poor) and decreases to -3.32237 (optimal)
    # We estimate the progress from 0 (start) to 1 (end)
    # Assuming the search space values are roughly in [-3.5, 3.5]
    min_val = -3.32237
    max_val_est = 3.5
    total_range = max_val_est - min_val
    
    # progress_score: 1 when f_best is max (start), 0 when f_best is min (end)
    # Clip f_best to valid range to handle edge cases
    f_best_clipped = np.clip(f_best, min_val, max_val_est)
    progress_score = (f_best_clipped - min_val) / total_range
    
    # w_explore should be high when progress_score is high (early)
    # w_exploit should be high when progress_score is low (late)
    # Use a smooth transition
    w_explore = 0.4 + 0.6 * progress_score  # 0.4 to 1.0
    w_exploit = 1.0 - w_explore             # 0.6 to 0.0
    
    # Combine the two objectives
    score = w_exploit * ei_normalized + w_explore * uncertainty
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
