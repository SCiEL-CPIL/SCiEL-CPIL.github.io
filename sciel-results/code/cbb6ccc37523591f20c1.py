"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Thompson Sampling acquisition with cost normalization and deadline guard.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions in objective units.
    sigma : array-like, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best (lowest) observed value.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    n = len(mu)
    
    # Thompson Sampling: draw a single sample from the posterior
    # sample = mu - sigma * z, where z ~ N(0,1)
    # We want to minimize, so we sample from the posterior and pick the minimum
    # Normalize by cost for cost-awareness
    
    # Use a single random draw for efficiency
    z = np.random.standard_normal(n)
    samples = mu - sigma * z  # Sample from posterior (minimization)
    
    # Cost-normalized score: we want high scores for points that are
    # likely to be good (low objective) relative to their cost
    # score = -sample / cost (higher is better since we minimize)
    scores = -samples / cost
    
    # Handle any non-finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
