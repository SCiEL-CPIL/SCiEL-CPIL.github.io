"""Rewritten acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of styblinski tang 1d.
    
    Uses Knowledge Gradient (KG) acquisition with Monte Carlo approximation.
    KG estimates the expected improvement in the global minimum by sampling
    from the posterior at top-k candidates, capturing multimodal structure
    better than pointwise EI or UCB.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : array_like, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed value so far (minimum).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores for each candidate. Higher is better.
    """
    start_time = time.time()
    deadline = start_time + 320.0  # Safety margin before 330s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = len(mu)
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # First, compute pointwise EI as a baseline for ranking
    z = (f_best - mu) / sigma
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Select top-k candidates by EI for KG sampling
    k = min(50, n)
    top_k_indices = np.argsort(ei)[-k:]
    
    # Compute KG for each candidate using Monte Carlo
    # For each candidate i, sample from its posterior and compute
    # E[min(f(x_i), f_best) | posterior]
    # KG(i) = E[f_best - min(f(x_i), f_best)] = E[max(f_best - f(x_i), 0)]
    # But this is just EI. The true KG considers correlation with other points.
    
    # Approximate KG: for each candidate, sample the posterior at top-k points
    # and compute the probability that this candidate is the new minimum
    n_samples = 500
    rng = np.random.default_rng(42)
    
    # Precompute samples for top-k candidates
    # Shape: (n_samples, k)
    samples = np.zeros((n_samples, k))
    for j in range(k):
        idx = top_k_indices[j]
        samples[:, j] = rng.normal(mu[idx], sigma[idx], size=n_samples)
    
    # For each candidate i in top-k, compute KG
    kg_scores = np.zeros(n)
    kg_scores[top_k_indices] = ei[top_k_indices]  # Initialize with EI
    
    # Compute KG for top-k candidates
    # KG(i) = E[f_best - min(f(x_i), min_{j!=i} f(x_j), f_best)]
    # Approximate: for each sample, compute the minimum across all top-k
    # and credit the improvement to the candidate that achieves it
    
    # Vectorized approach: for each sample, find which candidate gives the minimum
    # and accumulate the improvement
    for s in range(n_samples):
        if time.time() > deadline:
            break
        sample_values = samples[s, :]
        # Include f_best as a baseline
        all_values = np.concatenate([sample_values, [f_best]])
        min_val = np.min(all_values)
        improvement = f_best - min_val
        if improvement > 0:
            # Credit the improvement to the candidate that achieved the minimum
            min_idx = np.argmin(all_values)
            if min_idx < k:
                # This candidate achieved the new minimum
                # Add a fraction of the improvement to its KG score
                kg_scores[top_k_indices[min_idx]] += improvement / n_samples
    
    # Combine KG with EI for robustness
    # Use a blend: primarily KG for top candidates, EI for others
    scores = np.zeros(n)
    for idx in top_k_indices:
        # Blend KG and EI
        scores[idx] = 0.7 * kg_scores[idx] + 0.3 * ei[idx]
    scores[~np.isin(np.arange(n), top_k_indices)] = ei[~np.isin(np.arange(n), top_k_indices)]
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
