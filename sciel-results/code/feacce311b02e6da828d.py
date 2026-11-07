"""Cost-aware LCB acquisition for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 290.0  # safety margin before 300s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Lower Confidence Bound (LCB) acquisition normalized by cost.
    
    mu: array of shape (2000,) - GP mean predictions
    sigma: array of shape (2000,) - GP standard deviations
    f_best: float - best observed value so far
    cost: array of shape (2000,) - evaluation costs for each candidate
    
    Returns: array of shape (2000,) with scores (higher = evaluate next)
    """
    # Check deadline
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Return a safe fallback: just negative cost (prefer cheap candidates)
        return -cost.astype(np.float64)
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    n = len(mu)
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # Clip sigma to avoid division by zero and ensure positivity
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Dynamic exploration coefficient:
    # We want to balance exploration (uncertainty) and exploitation (low mean).
    # A common approach is to use a beta parameter that adapts.
    # Here we use a simple heuristic: as f_best approaches 0 (the minimum),
    # we can afford to be more exploitative. But since we don't know the true minimum
    # relative scale easily, let's use a fixed but reasonable beta.
    # 
    # For LCB: score = mu - beta * sigma
    # Then normalize by cost: score / cost
    #
    # The challenge is choosing beta. If beta is too large, we explore too much.
    # If too small, we exploit too much.
    #
    # Let's use an adaptive beta based on the spread of mu and sigma.
    # A reasonable approach: beta = sqrt(2 * log(1 + n)) / something, or just a tuned constant.
    # Given the problem structure (2D, [-5,5]^2, Matern-5/2 with length scale 0.2),
    # let's try a moderate beta.
    
    # Adaptive beta: scale with the typical magnitude of sigma
    # This helps when sigma varies a lot
    sigma_mean = np.mean(sigma_safe)
    if sigma_mean < 1e-6:
        # Very low uncertainty, be exploitative
        beta = 0.5
    else:
        # Moderate exploration
        beta = 1.5
    
    # Compute LCB score
    # Lower is better for the objective, so we want low mu and high sigma (uncertainty)
    # LCB: mu - beta * sigma (lower = more promising for minimization)
    lcb = mu - beta * sigma_safe
    
    # Normalize by cost: we want high scores for candidates with low LCB and low cost
    # score = -lcb / cost (negative LCB because lower LCB is better, divide by cost)
    # To ensure finite scores, clip cost away from zero
    cost_safe = np.maximum(cost, 1e-8)
    scores = -lcb / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
