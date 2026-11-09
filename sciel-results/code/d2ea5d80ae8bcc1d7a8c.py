# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Knowledge Gradient acquisition function.
    
    Approximates the expected improvement in the global minimum by
    using a Monte Carlo estimate of the posterior predictive distribution
    at the top-k candidates.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Mean of the GP posterior at candidate points
    sigma : array of shape (2000,)
        Standard deviation of the GP posterior at candidate points
    f_best : float
        Best observed value so far
        
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores, higher means more promising
    """
    n = len(mu)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Select top-k candidates based on a simple heuristic (low mu + high sigma)
    # This helps focus the KG computation on promising regions
    # Use a combination of mean and uncertainty to select candidates
    k = min(100, n)
    
    # Score candidates for selection: we want low mu and high sigma
    # Use a simple UCB-like score for selection
    selection_score = mu - 1.0 * sigma_safe
    top_k_indices = np.argsort(selection_score)[:k]
    
    # Initialize scores with EI for all candidates
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    scores = ei.copy()
    
    # For the top-k candidates, compute a more accurate KG using MC
    # We'll assume independence for the MC samples
    
    # Use more MC samples for better accuracy
    n_mc = 500
    rng = np.random.default_rng(42)
    
    # Pre-compute values for top-k
    mu_top = mu[top_k_indices]
    sigma_top = sigma_safe[top_k_indices]
    
    # Sample from each top-k candidate
    # Shape: (n_mc, k)
    mc_samples = rng.normal(loc=mu_top, scale=sigma_top, size=(n_mc, k))
    
    # For each top-k candidate i, compute KG
    # KG(i) = E[max(0, f_best - min(f_i, min_other))]
    # where min_other is the minimum of all other top-k candidates
    
    # Pre-compute min of all candidates for each MC sample
    min_all = np.min(mc_samples, axis=1)  # shape (n_mc,)
    
    # For each candidate, we need min of all others
    # Efficient approach: for each MC sample, find the two smallest values
    # If candidate i is the smallest, min_other is the second smallest
    # Otherwise, min_other is the smallest
    
    # Get indices of smallest and second smallest for each MC sample
    # Use argsort on each row
    sorted_indices = np.argsort(mc_samples, axis=1)  # shape (n_mc, k)
    min_idx = sorted_indices[:, 0]  # index of min in each sample
    second_min_idx = sorted_indices[:, 1]  # index of second min
    
    # For each candidate idx, compute KG
    for idx in range(k):
        # f_i is the sample for candidate idx
        f_i = mc_samples[:, idx]
        
        # For each MC sample, determine min_other
        # If idx is the min, min_other is the second min
        # Otherwise, min_other is the min
        is_min = (min_idx == idx)
        min_other = np.where(is_min, mc_samples[np.arange(n_mc), second_min_idx], min_all)
        
        # min(f_i, min_other)
        min_val = np.minimum(f_i, min_other)
        
        # max(0, f_best - min_val)
        kg_val = np.maximum(0, f_best - min_val)
        
        # Average over MC samples
        scores[top_k_indices[idx]] = np.mean(kg_val)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
