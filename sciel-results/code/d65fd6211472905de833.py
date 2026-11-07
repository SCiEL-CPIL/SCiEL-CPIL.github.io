"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _cost_normalized_ucb(mu, sigma, f_best, cost, beta=1.0):
    """
    Cost-normalized UCB acquisition.
    Higher scores are better (we want to minimize the objective).
    For minimization, we want low mu and high sigma.
    UCB for minimization: mu - beta * sigma
    Normalize by cost to prefer cheaper evaluations.
    """
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # UCB for minimization: we want to minimize mu - beta * sigma
    # So the score should be higher for lower (mu - beta * sigma)
    # Let's use: score = (f_best - mu + beta * sigma) / cost
    # This way, low mu (good) and high sigma (exploration) give high score.
    # Dividing by cost penalizes expensive evaluations.
    
    ucb = f_best - mu + beta * sigma_safe
    score = ucb / np.maximum(cost, 1e-8)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score


def _cost_normalized_ei(mu, sigma, f_best, cost, xi=0.0):
    """
    Cost-normalized Expected Improvement.
    """
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # EI for minimization
    imp = f_best - mu - xi
    z = imp / sigma_safe
    ei = imp * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Normalize by cost
    score = ei / np.maximum(cost, 1e-8)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score


def _hybrid_acquisition(mu, sigma, f_best, cost):
    """
    Hybrid acquisition combining UCB and EI with adaptive weighting.
    Uses the magnitude of sigma to decide between exploration and exploitation.
    """
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute both scores
    ucb_score = _cost_normalized_ucb(mu, sigma, f_best, cost, beta=1.0)
    ei_score = _cost_normalized_ei(mu, sigma, f_best, cost, xi=0.0)
    
    # Adaptive weighting: use more UCB when uncertainty is high
    # Normalize sigma to [0, 1] for weighting
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-8:
        sigma_norm = sigma_safe / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # Weight: more UCB when sigma is high, more EI when sigma is low
    w_ucb = 0.5 + 0.5 * sigma_norm
    w_ei = 1.0 - w_ucb
    
    score = w_ucb * ucb_score + w_ei * ei_score
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean predictions from the GP.
    sigma : ndarray, shape (2000,)
        Standard deviation predictions from the GP.
    f_best : float
        Best observed function value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    start_time = time.time()
    deadline = 280.0  # Safety margin before 300s limit
    
    # Validate inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    if mu.shape != (2000,) or sigma.shape != (2000,) or cost.shape != (2000,):
        raise ValueError("Input shapes must be (2000,)")
    
    # Handle edge cases
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)) or not np.all(np.isfinite(cost)):
        # Return uniform scores if inputs contain non-finite values
        return np.ones(2000, dtype=np.float64)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0.0)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-8)
    
    # Use hybrid acquisition
    scores = _hybrid_acquisition(mu, sigma, f_best, cost)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Return best result found with safety margin
        # Just return the current scores
        pass
    
    # Final validation: ensure all scores are finite
    if not np.all(np.isfinite(scores)):
        # Replace non-finite values with a reasonable default
        finite_mask = np.isfinite(scores)
        if np.any(finite_mask):
            default_val = np.median(scores[finite_mask])
        else:
            default_val = 0.0
        scores = np.where(np.isfinite(scores), scores, default_val)
    
    return scores
# EVOLVE-BLOCK-END
