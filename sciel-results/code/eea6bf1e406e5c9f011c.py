import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Knowledge Gradient (KG) approximation for minimization.
    We approximate the global minimum reference value for each candidate by considering
    the minimum predictive mean of the *other* candidates (the 'rest of the function').
    This provides a more informed baseline than just the current best observed value.
    """
    n = len(mu)
    
    # Find the indices of the smallest and second smallest means
    # Sorting is O(N log N), which is efficient for N=2000
    sorted_idx = np.argsort(mu)
    min_idx_1 = sorted_idx[0]
    min_idx_2 = sorted_idx[1] if n > 1 else 0
    
    mu_min_1 = mu[min_idx_1]
    mu_min_2 = mu[min_idx_2]
    
    # Determine the reference value for the "rest of the function" for each point
    # If we evaluate point i, the best value in the rest of the domain is approximated by:
    # - mu_min_2 if i is the global minimum (mu_min_1)
    # - mu_min_1 otherwise
    rest_mu = np.full(n, mu_min_1)
    rest_mu[min_idx_1] = mu_min_2
    
    # The effective reference for the new best is min(f_best, rest_mu)
    # We take the minimum because the new global best will be the lowest of:
    # 1. The current best observed (f_best)
    # 2. The value at the new point (Y_x)
    # 3. The best value in the rest of the domain (approximated by rest_mu)
    ref_vals = np.minimum(f_best, rest_mu)
    
    # Compute Expected Improvement (1-step KG) relative to ref_vals
    # EI = (ref_val - mu) * Phi((ref_val - mu) / sigma) + sigma * phi((ref_val - mu) / sigma)
    
    # Handle cases where sigma is very small to avoid division by zero
    safe_sigma = np.maximum(sigma, 1e-10)
    
    # Standardized distance from ref_val
    z = (ref_vals - mu) / safe_sigma
    
    # Expected improvement formula
    ei = (ref_vals - mu) * norm.cdf(z) + safe_sigma * norm.pdf(z)
    
    # Clip negative values to zero (EI should be non-negative)
    ei = np.maximum(ei, 0.0)
    
    # Return EI as the score (higher is better)
    scores = ei
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
