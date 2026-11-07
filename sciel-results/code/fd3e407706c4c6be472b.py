"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Compute a cost-aware acquisition score for 2000 candidate points.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviations
    f_best : float - best observed value so far
    cost : array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = better)
    """
    # Track elapsed time for safety
    t0 = time.time()
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    
    # Avoid division by zero in sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Standard EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip negative EI to zero (EI should be non-negative)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI: divide by cost
    cost_safe = np.maximum(cost, 1e-9)
    ei_cost = ei / cost_safe
    
    # UCB component for exploration
    # Use a beta that balances exploration and exploitation
    # In high dimensions, we need more exploration
    beta = 2.0
    ucb = mu + beta * sigma_safe
    
    # For minimization, we want low values, so UCB for minimization is:
    # We want to explore where mu is low AND sigma is high
    # A common approach: use -mu + beta * sigma for minimization
    # Or use EI which already handles this
    
    # Let's use a blend: primarily cost-aware EI with some UCB exploration
    # Normalize both to [0, 1] range for blending
    
    # Normalize EI cost
    ei_min = np.min(ei_cost)
    ei_max = np.max(ei_cost)
    ei_range = ei_max - ei_min + 1e-9
    ei_norm = (ei_cost - ei_min) / ei_range
    
    # For UCB in minimization, we want to reward low mu and high sigma
    # ucb_min = mu - beta * sigma (this is what we'd expect if we explore)
    ucb_min = mu - beta * sigma_safe
    ucb_min_cost = -ucb_min / cost_safe  # Negate because lower is better, and divide by cost
    
    ucb_min_min = np.min(ucb_min_cost)
    ucb_min_max = np.max(ucb_min_cost)
    ucb_min_range = ucb_min_max - ucb_min_min + 1e-9
    ucb_norm = (ucb_min_cost - ucb_min_min) / ucb_min_range
    
    # Blend: 70% EI, 30% UCB for exploration
    weight_ei = 0.7
    weight_ucb = 0.3
    
    scores = weight_ei * ei_norm + weight_ucb * ucb_norm
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Time check
    elapsed = time.time() - t0
    if elapsed > 280:  # Safety margin before 300s limit
        return scores
    
    return scores
# EVOLVE-BLOCK-END
