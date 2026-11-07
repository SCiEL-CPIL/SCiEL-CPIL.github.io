import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a Knowledge Gradient (KG) approximation.
    
    This implementation uses an independence approximation for the KG.
    It estimates the expected reduction in the global minimum by comparing the 
    candidate point against an estimate of the minimum of the *other* points.
    
    Steps:
    1. Estimate the "rest of the search space" minimum (mu_rest) for each point.
       - For most points, this is the global minimum of the means.
       - For the point with the global minimum mean, this is the second lowest mean.
    2. The effective threshold for improvement is K_i = min(f_best, mu_rest[i]).
    3. Compute the expected improvement of the point Y_i over this threshold K_i.
       E[min(K_i, Y_i)] is derived analytically.
       KG_i = K_i - E[min(K_i, Y_i)]
    """
    n = len(mu)
    
    # Handle edge cases
    if n == 0:
        return np.array([])
    
    # Ensure sigma is positive for division
    safe_sigma = np.maximum(sigma, 1e-10)
    
    # Find the indices of the two smallest predictive means
    # We need these to approximate mu_rest (the expected min of the other points)
    if n == 1:
        # If only one point, the "rest" is effectively f_best (or infinity if no prior, but f_best exists)
        # Actually, if n=1, there are no "other" points in the candidate set.
        # The KG falls back to EI against f_best.
        mu_rest = np.full(n, f_best)
    else:
        # Get indices of smallest values
        # argsort is O(N log N), acceptable for 2000
        sorted_indices = np.argsort(mu)
        min_idx_1 = sorted_indices[0]
        min_idx_2 = sorted_indices[1]
        
        # Initialize mu_rest with the global minimum mean
        # This assumes that for any point i != min_idx_1, the best of the others is mu[min_idx_1]
        mu_rest = np.full(n, mu[min_idx_1])
        
        # For the point that IS the global minimum, the best of the others is mu[min_idx_2]
        mu_rest[min_idx_1] = mu[min_idx_2]
        
        # If n > 2, this is a good approximation.
        # If n == 2, it is exact for the independence approximation.
        
    # Calculate the effective threshold K_i
    # The minimum can be no lower than min(f_best, expected_min_of_rest)
    K = np.minimum(f_best, mu_rest)
    
    # Standardized distance z
    z = (K - mu) / safe_sigma
    
    # Calculate the KG score
    # KG_i = K_i - E[min(K_i, Y_i)]
    # E[min(K, Y)] = K + (mu - K)*Phi(z) - sigma*phi(z)
    # KG_i = -(mu - K)*Phi(z) + sigma*phi(z)
    # KG_i = (K - mu)*Phi(z) + sigma*phi(z)
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    scores = (K - mu) * Phi_z + safe_sigma * phi_z
    
    # Due to floating point errors, scores might be slightly negative
    # Theoretically, KG >= 0
    scores = np.maximum(scores, 0.0)
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
