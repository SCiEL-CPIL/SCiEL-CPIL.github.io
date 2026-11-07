"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions
    sigma : array-like, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    array of shape (2000,) with finite scores (higher = evaluate next)
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute z-score for expected improvement
    z = (f_best - mu) / sigma
    
    # Expected improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Adaptive beta: shift from exploration to exploitation as f_best improves
    # For rosenbrock, minimum is 0, so as f_best approaches 0, we exploit more
    # f_best starts around 100+ and decreases toward 0
    # Use a smooth transition: beta = 0.8 + 1.2 * sigmoid(-f_best / scale)
    # When f_best is large (early), sigmoid(-large) ~ 0, beta ~ 0.8 (exploitation)
    # When f_best is small (late), sigmoid(-small) ~ 1, beta ~ 2.0 (exploration)
    # Actually, we want MORE exploration early, so let's reverse:
    # beta = 0.8 + 1.2 * sigmoid(f_best / scale)
    # When f_best is large (early), sigmoid(large) ~ 1, beta ~ 2.0 (exploration)
    # When f_best is small (late), sigmoid(small) ~ 0, beta ~ 0.8 (exploitation)
    
    scale = 10.0  # Scale parameter for the transition
    beta = 0.8 + 1.2 / (1.0 + np.exp(-f_best / scale))
    
    # Cost-aware: divide by cost
    scores = ei / (cost + 1e-9)
    
    # Blend with uncertainty for exploration
    # Higher beta means more weight on uncertainty
    uncertainty_term = sigma / (sigma.mean() + 1e-9)
    scores = scores * (1.0 - beta / 3.0) + uncertainty_term * (beta / 3.0)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
