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
    
    # Select top-k candidates based on a combination of low mu and high sigma
    # Use a larger k for better coverage of promising regions
    k = min(100, n)
    
    # Score candidates for selection: we want low mu and high sigma
    # Use a UCB-like score for selection with tuned beta
    beta = 1.5
    selection_score = mu - beta * sigma_safe
    top_k_indices = np.argsort(selection_score)[:k]
    
    # Pre-compute some values
    mu_top = mu[top_k_indices]
    sigma_top = sigma_safe[top_k_indices]
    
    # Initialize scores
    scores = np.zeros(n)
    
    # Compute EI for all candidates as a baseline
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # For the top-k candidates, compute KG using Monte Carlo
    # Use more MC samples for better accuracy
    n_mc = 200
    
    # Pre-generate MC samples for the top-k candidates
    # For each of the n_mc samples, draw from each top-k candidate
    rng = np.random.default_rng(42)
    
    # Sample from each top-k candidate
    # Shape: (n_mc, k)
    mc_samples = rng.normal(loc=mu_top, scale=sigma_top, size=(n_mc, k))
    
    # For each top-k candidate i, compute KG
    # KG(i) = E[max(0, f_best - min(f_i, f_j for j in top-k, j != i))]
    # We can vectorize this computation
    for idx in range(k):
        # Get the sample for candidate i
        f_i = mc_samples[:, idx]  # shape (n_mc,)
        
        # Get the min of all other top-k candidates
        # Create a mask to exclude idx
        mask = np.ones(k, dtype=bool)
        mask[idx] = False
        min_other = np.min(mc_samples[:, mask], axis=1)  # shape (n_mc,)
        
        # min(f_i, min_other)
        min_val = np.minimum(f_i, min_other)
        
        # max(0, f_best - min_val)
        kg_val = np.maximum(0, f_best - min_val)
        
        # Average over MC samples
        scores[top_k_indices[idx]] = np.mean(kg_val)
    
    # For candidates not in top-k, use EI as a reasonable proxy
    mask_not_topk = np.ones(n, dtype=bool)
    mask_not_topk[top_k_indices] = False
    
    # Use EI for non-top-k candidates
    scores[mask_not_topk] = ei[mask_not_topk]
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
