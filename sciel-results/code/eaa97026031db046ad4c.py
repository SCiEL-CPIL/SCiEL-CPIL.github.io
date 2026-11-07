"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a combination of:
    1. Expected Improvement (EI) per unit cost
    2. Knowledge gradient-like term for exploration
    3. Penalty for high cost regions
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array of shape (2000,)
        Cost of evaluating each point
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = more desirable)
    """
    # Deadline guard: track elapsed time to ensure we finish in time
    t_start = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score for expected improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware: divide by cost to get EI per unit cost
    cost_safe = np.maximum(cost, 1e-9)
    ei_per_cost = ei / cost_safe
    
    # Add exploration bonus based on uncertainty
    # Points with high sigma relative to the range of mu are more exploratory
    mu_range = np.ptp(mu) if np.ptp(mu) > 1e-9 else 1.0
    exploration_bonus = sigma_safe / (mu_range + 1e-9)
    
    # Balance exploitation (EI per cost) and exploration
    # Use a weighted combination
    alpha = 0.1  # Exploration weight
    
    scores = ei_per_cost + alpha * exploration_bonus
    
    # Add small penalty for very high cost points to prefer cheaper evaluations
    # when EI is similar
    cost_penalty = 0.01 * (cost_safe - cost_safe.mean()) / (cost_safe.std() + 1e-9)
    scores = scores - cost_penalty
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - t_start
    if elapsed > 280:  # Safety margin before 300s limit
        return scores
    
    return scores
# EVOLVE-BLOCK-END
