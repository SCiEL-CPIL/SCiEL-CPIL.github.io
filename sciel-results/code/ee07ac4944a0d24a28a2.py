"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Parameters:
    - mu: array of shape (2000,) - GP mean predictions
    - sigma: array of shape (2000,) - GP standard deviations
    - f_best: float - best observed value so far
    - cost: array of shape (2000,) - cost of evaluating each point
    
    Returns:
    - scores: array of shape (2000,) - acquisition values (higher = prefer)
    """
    # Deadline guard
    t_start = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: replace non-finite values
    # If mu is not finite, use f_best (conservative)
    mu = np.where(np.isfinite(mu), mu, f_best)
    # If sigma is not finite or negative, use a small value to avoid NaNs in sampling
    # However, if sigma is 0, we just use mu.
    sigma = np.where(np.isfinite(sigma) & (sigma >= 0), sigma, 0.0)
    # Cost must be positive. If not, use 1.0
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Thompson Sampling: Sample from the posterior
    # y ~ N(mu, sigma^2)
    # To avoid issues with very large sigma, we can clip sigma, but TS is robust.
    # Ensure sigma is not negative for sqrt in other contexts, but here we just multiply.
    
    # Draw samples
    # Using standard normal
    noise = np.random.randn(mu.shape[0])
    samples = mu + sigma * noise
    
    # Cost-aware scoring:
    # We want to minimize f(x) / cost(x)
    # Since we return scores where HIGHER is better, we return - (f(x) / cost(x))
    
    # Handle division by zero or non-finite samples
    # If sample is not finite, assign a very bad score (very negative)
    # If cost is 0 (handled above to be >= 1), no div by zero.
    
    ratios = samples / cost
    
    # If a sample is non-finite (e.g. inf), the ratio is inf.
    # -inf is the lowest possible score, which is correct (we don't want to pick it).
    # However, if samples contain NaN, ratios contain NaN.
    # We need to ensure returned scores are finite.
    
    # Replace non-finite ratios with a large positive number (so -ratio is large negative)
    # Or simply replace non-finite samples with f_best before ratio?
    # Better: compute ratios, then fix non-finite scores.
    
    scores = -ratios
    
    # Ensure scores are finite
    # If score is NaN or Inf, replace with a very negative number (worst case)
    # Using -1e30 as a very bad score
    scores = np.where(np.isfinite(scores), scores, -1e30)
    
    # Deadline check (should be instantaneous, but just in case)
    if time.time() - t_start > 0.1: # 100ms limit for this function
         # Return a fallback, e.g. based on mu only
         fallback_scores = -mu / cost
         fallback_scores = np.where(np.isfinite(fallback_scores), fallback_scores, -1e30)
         return fallback_scores

    return scores
# EVOLVE-BLOCK-END
