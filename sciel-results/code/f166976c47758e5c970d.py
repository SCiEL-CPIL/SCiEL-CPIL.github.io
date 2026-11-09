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
    
    # Compute EI for all candidates to use for selection and non-top-k scoring
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Select top-k candidates based on EI
    # A larger k ensures we don't miss promising points
    k = min(100, n)
    top_k_indices = np.argsort(ei)[::-1][:k]
    
    # Initialize scores with EI for all points
    scores = ei.copy()
    
    # If k is small, just return EI (KG is expensive for very small k? No, KG is better)
    # But if k == 1, KG is just EI.
    if k <= 1:
        return scores
    
    # Monte Carlo parameters
    n_mc = 200
    
    # Pre-compute values for top-k
    mu_top = mu[top_k_indices]
    sigma_top = sigma_safe[top_k_indices]
    
    # Generate MC samples for all top-k candidates
    # Shape: (n_mc, k)
    rng = np.random.default_rng(42)
    mc_samples = rng.normal(loc=mu_top, scale=sigma_top, size=(n_mc, k))
    
    # Compute KG for each top-k candidate
    # We want KG_i = E[max(0, f_best - min(f_i, min_{j != i} f_j))]
    
    # Optimization: 
    # For each i, we need min of all columns except i.
    # We can compute the min of the entire array, and if the min is at i, we need the second min.
    # However, doing this per i in a vectorized way is tricky.
    # The loop approach is O(k * n_mc * k). With k=100, n_mc=200, that's 2,000,000 ops, which is fine.
    
    for idx in range(k):
        # Get samples for candidate i
        f_i = mc_samples[:, idx]
        
        # Get min of all other candidates
        # Create a mask or use slicing
        # To avoid copying the whole array, we can do:
        # min_other = np.min(np.delete(mc_samples, idx, axis=1), axis=1)
        # np.delete is a bit slow. 
        # Alternative: 
        # Compute global min and argmin.
        # If argmin == idx, then min_other is the min of the rest.
        # But we need this for every idx.
        
        # Let's stick to the copy/delete method for clarity, or a more efficient vectorized trick.
        # Efficient trick:
        # min1 = np.min(mc_samples, axis=1)
        # argmin1 = np.argmin(mc_samples, axis=1)
        # min2 = np.min(np.where(np.arange(k)[np.newaxis, :] == argmin1[:, np.newaxis], np.inf, mc_samples), axis=1)
        
        # This computes the second smallest value in each row.
        # If the smallest value is at index idx, then min_other for idx is min2.
        # If the smallest value is NOT at index idx, then min_other for idx is min1 (because the global min is among the "others").
        
        min1 = np.min(mc_samples, axis=1)
        argmin1 = np.argmin(mc_samples, axis=1)
        
        # Compute second minimum
        # Replace the minimum with inf and take min again
        mc_samples_copy = mc_samples.copy()
        mc_samples_copy[np.arange(n_mc), argmin1] = np.inf
        min2 = np.min(mc_samples_copy, axis=1)
        
        # For candidate idx:
        # If the global minimum (min1) is at idx, then the min of others is min2.
        # If the global minimum is NOT at idx, then the min of others is min1 (since min1 is from some other j).
        
        is_min_at_idx = (argmin1 == idx)
        
        min_other = np.where(is_min_at_idx, min2, min1)
        
        # Now compute KG
        min_val = np.minimum(f_i, min_other)
        kg_val = np.maximum(0, f_best - min_val)
        
        scores[top_k_indices[idx]] = np.mean(kg_val)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
