import numpy as np
import time
from scipy.stats import norm

# Module-level time tracking for deadline guard
_start_time = time.time()
_DEADLINE_MARGIN = 15.0  # seconds before the 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Hartmann 3D.
    Uses Thompson Sampling with cost-awareness.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions at 2000 candidate points
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions at 2000 candidate points
    f_best : float
        Best (lowest) observed function value so far
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point (1 + mean of coordinates scaled to [0,1])
    
    Returns:
    --------
    score : ndarray, shape (2000,)
        Acquisition scores; higher values are preferred for next evaluation
    """
    # Deadline guard: if we're getting close to the 330s limit, return quickly
    elapsed = time.time() - _start_time
    if elapsed > 330.0 - _DEADLINE_MARGIN:
        # Emergency: just return a simple score to avoid timeout
        return np.full(2000, 0.0)
    
    # Handle edge cases
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):
        mu = np.where(np.isfinite(mu), mu, 0.0)
        sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    
    if not np.all(np.isfinite(cost)) or np.any(cost <= 0):
        cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-10)
    
    # Thompson Sampling: draw a sample from N(mu, sigma) for each candidate
    # We want to minimize, so lower sampled values are better
    # But we return scores where HIGHER = evaluate next
    # So we negate the sampled value (or use negative for minimization)
    
    # Draw a single random sample for each candidate
    # Use a fixed seed based on f_best and some entropy to make it deterministic per call
    # Actually, we just use np.random for stochasticity
    rng = np.random.default_rng()
    
    # Sample from the GP posterior
    sampled = mu + sigma * rng.standard_normal(len(mu))
    
    # For minimization, we want to find the minimum
    # The score should be high for points we want to evaluate
    # So we use negative of the sampled value (lower sampled = higher score)
    ts_score = -sampled
    
    # Cost-awareness: divide by cost to prefer cheaper evaluations
    # But we need to be careful: if we just divide, we might over-penalize
    # A better approach: use a cost-effective version
    # CEI-style: score = ts_score / cost
    # This naturally balances exploration (high sigma) with cost
    
    score = ts_score / cost
    
    # Safety: ensure finite values
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score

def _norm_cdf(x):
    """Standard normal CDF using scipy"""
    return norm.cdf(x)
