"""Acquisition function for Bayesian optimization of a fixed seeded random RBF mixture."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Start time for deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 315  # 315s deadline, 15s margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian Optimization.
    Uses a hybrid strategy combining Expected Improvement (EI) with
    uncertainty-based exploration to balance exploitation and exploration.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: return random scores to avoid timeout
        return np.random.randn(len(mu))

    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Compute Expected Improvement (EI) for minimization
    # EI = E[max(f_best - f(x), 0)]
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Compute uncertainty-based exploration bonus
    # This encourages exploration in high-uncertainty regions
    # Use a normalized version of sigma to avoid scale issues
    sigma_mean = np.mean(sigma)
    sigma_std = np.std(sigma)
    if sigma_std > 1e-8:
        sigma_norm = (sigma - sigma_mean) / sigma_std
    else:
        sigma_norm = np.zeros_like(sigma)
    
    # Combine EI and exploration bonus
    # The weight on exploration should be moderate to balance exploitation
    exploration_weight = 0.5
    
    scores = ei + exploration_weight * sigma_norm
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
