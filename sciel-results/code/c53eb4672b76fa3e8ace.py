"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Quantile-based probability of improvement (PI) acquisition, normalized by cost^2.
    
    This favors candidates with high confidence of beating f_best while heavily
    penalizing expensive evaluations to preserve budget for high-uncertainty regions.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - lowest observed value so far
    cost : array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - finite scores (higher = evaluate next)
    """
    # Ensure we have finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score: how many std devs below f_best is the mean
    z = (f_best - mu) / sigma_safe
    
    # Probability of improvement: P(f(x) < f_best) = Phi(z)
    # This is a quantile-based measure - it directly gives the probability
    # that the true function value at x is better than f_best
    pi = norm.cdf(z)
    
    # We want to favor candidates with high PI, but also account for uncertainty.
    # A pure PI can be too conservative. Let's blend with a small exploration term.
    # Use a softplus-like transformation to keep values positive and smooth.
    # Actually, let's stick closer to the directive: PI normalized by cost^2.
    
    # Cost normalization: divide by cost^2 to heavily penalize expensive points
    # cost is in [1, 2] since it's 1 + mean(x scaled to [0,1])
    cost_safe = np.maximum(cost, 1e-9)
    cost_sq = cost_safe ** 2
    
    # Base score: PI / cost^2
    scores = pi / cost_sq
    
    # Add a small exploration bonus for high uncertainty regions
    # This helps escape local optima by giving some credit to uncertain areas
    # Weight it so it doesn't dominate the PI term
    exploration_bonus = 0.05 * sigma_safe / cost_sq
    
    scores = scores + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure non-negative (for safety, though PI is already in [0,1])
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
