"""Cost-aware acquisition function for Bayesian optimization (three hump camel 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Args:
        mu: (N,) array of GP mean predictions.
        sigma: (N,) array of GP standard deviations.
        f_best: float, best observed value so far.
        cost: (N,) array of costs for each candidate.
    
    Returns:
        (N,) array of acquisition scores (higher is better).
    """
    N = len(mu)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Standardized distance
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: For minimization, EI is usually defined as E[max(f_best - f(x), 0)]
    # If f(x) ~ N(mu, sigma^2), then E[max(f_best - f(x), 0)] = (f_best - mu)Phi(z) + sigma phi(z)
    # where z = (f_best - mu) / sigma
    ei_term = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei_term = np.maximum(ei_term, 0.0)
    
    # Adaptive beta for exploration (UCB-like term)
    # We want to explore more when f_best is poor (large) and exploit when f_best is good (small, close to 0)
    # Using a logarithmic schedule as suggested by inspiration 1
    # beta = multiplier * log(1 + f_best)
    # If f_best is negative (which shouldn't happen if min is 0 and we start with random points, but just in case), 
    # we clip it to 0 for the log argument to avoid issues, or use abs.
    # Since the minimum is 0, f_best >= 0.
    f_best_safe = np.maximum(f_best, 0.0)
    beta = 2.0 * np.log(1.0 + f_best_safe)
    
    # Floor beta at a small value to ensure some exploration always? 
    # Inspiration 1 mentioned "Floor at 1.0". Let's try floor at 0.5 or 1.0.
    # If beta is too high, we might waste evaluations on high variance regions that are not promising.
    # If beta is too low, we might get stuck in local minima.
    # Let's use a floor of 0.5.
    beta = np.maximum(beta, 0.5)
    
    # Exploration term: beta * sigma
    exploration_term = beta * sigma_safe
    
    # Combined acquisition value before cost normalization
    # We maximize this.
    # Note: EI is already a measure of expected improvement.
    # Adding beta*sigma encourages picking points with high uncertainty.
    combined_value = ei_term + exploration_term
    
    # Cost-aware: divide by cost
    # Add small epsilon to cost to avoid division by zero
    cost_safe = np.maximum(cost, 1e-10)
    
    scores = combined_value / cost_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
