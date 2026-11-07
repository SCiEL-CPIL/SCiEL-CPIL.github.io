"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Weighted EI + Exploration.
    
    Combines Expected Improvement (EI) with a simple exploration bonus
    proportional to the standard deviation (sigma), normalized by
    evaluation cost. This avoids the instability of normalizing by
    f_best and provides a stable balance between exploitation and
    exploration.
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed objective value (lower is better).
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    np.ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-10)
    
    # Calculate z-score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues in norm.cdf/pdf
    z_clipped = np.clip(z, -10.0, 10.0)
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # Exploration term: simple bonus for high uncertainty
    # beta controls the trade-off. 0.5 is a reasonable starting point.
    # Since EI is in objective units, sigma is also in objective units.
    # We add a fraction of sigma to encourage exploration.
    beta_explore = 0.5
    exploration_bonus = beta_explore * sigma_safe
    
    # Combined acquisition value
    # Higher is better.
    # EI is positive when improvement is expected.
    # Exploration bonus is always positive, pushing towards high sigma.
    combined = ei + exploration_bonus
    
    # Cost-aware: divide by cost
    score = combined / cost_safe
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
