"""Combined cost-aware acquisition: EI per cost plus uncertainty-scaled exploration."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard
_DEADLINE = time.time() + 320.0  # safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    mu, sigma: shape (2000,) - GP mean and std (in objective units)
    f_best: scalar - lowest observed value
    cost: shape (2000,) - evaluation cost for each candidate
    
    Returns: shape (2000,) - scores (higher = evaluate next)
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against zero/negative costs
    cost = np.maximum(cost, 1e-9)
    
    # Guard against zero/negative sigma
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute delta = f_best - mu
    delta = f_best - mu
    
    # Compute z = delta / sigma
    z = delta / sigma
    
    # Expected Improvement (EI)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    ei = delta * Phi_z + sigma * phi_z
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: EI per unit cost (exploitation)
    ei_cost = ei / cost
    
    # Uncertainty-based exploration term: sigma / cost
    # Encourages exploration in high-uncertainty, low-cost regions
    ucb_term = sigma / cost
    
    # Adaptive weight: when average uncertainty is high (early), explore more
    # When uncertainty is low (late), rely more on EI
    sigma_mean = np.mean(sigma)
    sigma_max = np.max(sigma)
    # Normalize: ratio of mean to max gives a sense of how "spread" the uncertainty is
    # Early: sigma is large everywhere, ratio ~1, so w_explore is higher
    # Late: sigma is small, but relative spread matters
    # Use a simple scaling: w_explore between 0.3 and 1.0
    # If sigma_mean is large relative to f_best magnitude, we're early
    f_best_abs = abs(f_best) + 1e-9
    # Scale sigma_mean to [0, 1] range based on f_best
    # If sigma_mean > f_best_abs, we're very early
    early_indicator = np.clip(sigma_mean / (f_best_abs + 1e-9), 0.0, 1.0)
    w_explore = 0.3 + 0.7 * early_indicator
    
    score = ei_cost + w_explore * ucb_term
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
