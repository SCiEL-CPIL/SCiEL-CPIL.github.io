import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Computes a Look-Ahead acquisition score for Bayesian Optimization.
    This strategy simulates the effect of evaluating each candidate point on the
    best observed value (f_best) and estimates the expected improvement from
    the subsequent best candidate, effectively planning one step ahead.
    """
    N = len(mu)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # 1. Calculate immediate Expected Improvement (EI) for all points
    z_curr = (f_best - mu) / sigma
    ei_curr = (f_best - mu) * norm.cdf(z_curr) + sigma * norm.pdf(z_curr)
    
    # 2. Look-Ahead:
    # For each candidate i, assume we evaluate it.
    # Approximate the new observation as the mean mu[i] (conservative estimate).
    # New f_best' = min(f_best, mu[i])
    
    # We can vectorize this by creating a matrix where rows represent the 
    # hypothetical evaluation of point i, and columns represent the subsequent 
    # evaluation of point j.
    
    # f_best_prime for each hypothetical i
    f_best_prime = np.minimum(f_best, mu)  # Shape (N,)
    
    # Create matrix of z-scores for all pairs (i, j)
    # z[i, j] = (f_best_prime[i] - mu[j]) / sigma[j]
    # f_best_prime[:, None] has shape (N, 1)
    # mu[None, :] has shape (1, N)
    # sigma[None, :] has shape (1, N)
    
    z_matrix = (f_best_prime[:, None] - mu[None, :]) / sigma[None, :]
    
    # Calculate EI matrix
    # ei_matrix[i, j] = EI of point j given f_best_prime[i]
    ei_matrix = (f_best_prime[:, None] - mu[None, :]) * norm.cdf(z_matrix) + sigma[None, :] * norm.pdf(z_matrix)
    
    # We want the max EI for the next step, excluding the point we just evaluated (i)
    # Set diagonal to -inf so it's not selected
    np.fill_diagonal(ei_matrix, -np.inf)
    
    # Max EI for the next step for each i
    max_next_ei = np.max(ei_matrix, axis=1)
    
    # 3. Combine scores
    # We weight the immediate EI and the expected future EI.
    # A weight of 1.0 means we value the future improvement equally to the current one.
    # This encourages points that not only improve f_best now but also set up a good position for the next step.
    alpha = 1.0
    scores = ei_curr + alpha * max_next_ei
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
