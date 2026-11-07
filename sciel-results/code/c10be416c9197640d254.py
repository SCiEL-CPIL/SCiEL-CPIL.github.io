"""Rewritten acquisition for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function that combines Expected Improvement (EI) with a
    quantile-based exploration bonus to encourage systematic coverage of
    high-uncertainty regions while still exploiting promising areas.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        Posterior mean at candidate points (in objective units).
    sigma : array_like, shape (2000,)
        Posterior standard deviation at candidate points (in objective units).
    f_best : float
        Lowest observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Guard against zero or negative sigma
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Expected Improvement (minimization)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Quantile-based exploration bonus:
    # Partition candidates into bins by posterior mean, and within each bin,
    # assign a bonus based on the maximum sigma in that bin. This encourages
    # exploration in high-uncertainty regions while maintaining exploitation
    # via EI.
    n = len(mu)
    n_bins = 20  # Number of quantile bins
    
    # Sort by mu and assign bin indices
    mu_sorted = np.sort(mu)
    bin_edges = np.linspace(0, n, n_bins + 1).astype(int)
    bin_edges[-1] = n  # Ensure last bin includes all
    
    # Assign each candidate to a bin based on its mu value
    bin_indices = np.searchsorted(mu_sorted, mu, side='right') - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # For each bin, compute the max sigma
    max_sigma_per_bin = np.zeros(n_bins)
    for b in range(n_bins):
        mask = bin_indices == b
        if np.any(mask):
            max_sigma_per_bin[b] = np.max(sigma_safe[mask])
    
    # Assign exploration bonus: candidates get a bonus proportional to
    # the max sigma in their bin, normalized to be comparable to EI scale
    exploration_bonus = max_sigma_per_bin[bin_indices]
    
    # Combine EI and exploration bonus with a balanced weighting
    # The weight is chosen to balance exploitation and exploration
    weight = 0.5
    scores = ei + weight * exploration_bonus
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
