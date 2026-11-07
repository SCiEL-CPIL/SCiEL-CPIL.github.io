"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Knowledge Gradient (KG) acquisition function via Monte Carlo integration.
    Computes the expected reduction in the minimum observed value for each candidate,
    accounting for the joint distribution over all candidates to balance
    exploration and exploitation for multi-modal functions like Hartmann 3D.
    
    Parameters:
    -----------
    mu : array-like, shape (n_candidates,)
        Mean of the GP posterior at each candidate point.
    sigma : array-like, shape (n_candidates,)
        Standard deviation of the GP posterior at each candidate point.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (n_candidates,)
        KG scores for each candidate. Higher is better.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    n_candidates = mu.shape[0]
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Monte Carlo KG: sample all candidates jointly (independent approximation)
    # For each sample, compute the new best if we evaluated each candidate i,
    # i.e., min(f_i, f_best), and the improvement is f_best - min(f_i, f_best).
    # The KG for candidate i is the expected improvement over many samples.
    # 
    # To capture global structure, we also compute the expected minimum over ALL
    # candidates in each sample, and use this to weight the improvement:
    # candidates that are near the sample minimum get a boost, reflecting that
    # they are in promising basins.
    
    n_samples = 500
    rng = np.random.default_rng(2024)
    
    # Generate samples: (n_samples, n_candidates)
    samples = rng.normal(loc=mu, scale=sigma_safe, size=(n_samples, n_candidates))
    
    # For each sample, the minimum over all candidates
    sample_min = np.min(samples, axis=1)  # shape (n_samples,)
    
    # For each candidate i and each sample s, the improvement if we evaluate i:
    # improvement_{i,s} = max(f_best - samples[s, i], 0)
    improvements = np.maximum(f_best - samples, 0.0)  # shape (n_samples, n_candidates)
    
    # Base KG: expected improvement (this is just EI, computed via MC)
    kg_base = np.mean(improvements, axis=0)  # shape (n_candidates,)
    
    # Global basin indicator: for each candidate i, how close is it to the
    # sample minimum in each sample? This captures whether i is in a promising region.
    # Use a soft indicator: exp(-(samples - sample_min)^2 / (2 * tau^2))
    # where tau is a scale parameter.
    # To avoid underflow, use a relative measure.
    # Alternative: rank-based. For each sample, compute the rank of each candidate.
    # Candidates with low rank (near minimum) get higher weight.
    
    # Compute for each sample, the fraction of candidates that are worse than i
    # This gives a soft "probability of being near the minimum"
    # More efficient: for each sample, count how many candidates are <= samples[s,i]
    # This is O(n_samples * n_candidates^2) which is too slow for 2000 candidates.
    
    # Instead, use a simpler approach: for each candidate i, compute the probability
    # that it is the global minimum (as before), and use that to weight the KG.
    min_indices = np.argmin(samples, axis=1)
    counts = np.bincount(min_indices, minlength=n_candidates)
    prob_global = counts / n_samples  # shape (n_candidates,)
    
    # Also compute a second moment: the probability that i is among the top-k
    # candidates, to capture basin structure.
    k = 10
    # For each sample, get the k smallest indices
    # This is expensive, so approximate with: prob that samples[s,i] < sample_min + delta
    # where delta is the k-th smallest value minus the minimum.
    # Simpler: just use prob_global with a larger effective support.
    
    # Combine: KG score = base_KG * (1 + alpha * prob_global)
    # The prob_global term boosts candidates that are likely in the global basin,
    # helping to navigate multi-modal landscapes.
    alpha = 1.5
    scores = kg_base * (1.0 + alpha * prob_global)
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
