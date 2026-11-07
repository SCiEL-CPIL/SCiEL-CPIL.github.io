"""Thompson Sampling cost-aware acquisition for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    Samples a single function value from the GP posterior for each candidate
    and scores by minimizing the ratio of sampled value to cost.
    Higher scores indicate more promising candidates to evaluate next.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions for 2000 candidate points
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions for 2000 candidate points
    f_best : float
        Best (lowest) observed function value
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x_scaled))
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more promising to evaluate next)
    """
    # Explicit deadline guard
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 300s kill
    
    # Ensure inputs are numpy arrays of float64
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Sanitize inputs
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-9), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 1e-9), cost, 1e-9)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Thompson Sampling: draw one sample from N(0,1) for all candidates
    # This creates a single stochastic function value for each candidate
    z = np.random.normal(0.0, 1.0, size=mu.shape)
    
    # Sampled function values: y ~ N(mu, sigma^2)
    sampled_values = mu + sigma * z
    
    # Score: we want to minimize sampled_value / cost
    # So acquisition score is negative of that ratio (higher = better)
    # Using -sampled_value / cost means we prefer low function values and low cost
    scores = -sampled_values / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Time check
    if time.time() > deadline:
        return scores
        
    return scores
# EVOLVE-BLOCK-END
