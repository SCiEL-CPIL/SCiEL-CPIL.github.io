"""Acquisition function for Bayesian optimization of Weierstrass 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Diversity-penalized acquisition: combines EI with a penalty based on
    pairwise correlation among top candidates using Matern-5/2 kernel structure.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        Posterior mean at candidate points.
    sigma : ndarray, shape (2000,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best (lowest) observed function value.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition values; higher means evaluate next.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    n = len(mu)
    
    # Expected Improvement for minimization
    improvement = f_best - mu
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Normalize EI to [0, 1]
    ei_max = np.max(ei)
    ei_min = np.min(ei)
    ei_range = ei_max - ei_min
    if ei_range > 1e-9:
        ei_norm = (ei - ei_min) / ei_range
    else:
        ei_norm = np.ones_like(ei)
    
    # Compute a diversity penalty based on local density of high-EI candidates
    # Use the structure of Matern-5/2 kernel: k(r) = (1 + sqrt(3)*r) * exp(-sqrt(3)*r)
    # where r is distance in scaled space (length scale 0.2 on [0,1])
    
    # We don't have coordinates, but we can use sigma as a proxy for local scale
    # Candidates with similar (mu, sigma) are likely in similar regions
    # Use a simple density-based penalty
    
    # Select top-K candidates by EI score
    K = min(50, n)
    top_k_idx = np.argsort(ei_norm)[-K:]
    top_k_scores = ei_norm[top_k_idx]
    top_k_sigma = sigma_safe[top_k_idx]
    top_k_mu = mu[top_k_idx]
    
    # Compute pairwise "similarity" using normalized (mu, sigma) features
    # Scale mu and sigma to comparable ranges
    mu_range = np.max(top_k_mu) - np.min(top_k_mu)
    sigma_range = np.max(top_k_sigma) - np.min(top_k_sigma)
    
    if mu_range > 1e-9:
        mu_scaled = (top_k_mu - np.min(top_k_mu)) / mu_range
    else:
        mu_scaled = np.zeros_like(top_k_mu)
    
    if sigma_range > 1e-9:
        sigma_scaled = (top_k_sigma - np.min(top_k_sigma)) / sigma_range
    else:
        sigma_scaled = np.zeros_like(top_k_sigma)
    
    # Distance in feature space (proxy for kernel distance)
    # Use Euclidean distance in scaled (mu, sigma) space
    mu_diff = np.abs(mu_scaled[:, None] - mu_scaled[None, :])
    sigma_diff = np.abs(sigma_scaled[:, None] - sigma_scaled[None, :])
    dist = np.sqrt(mu_diff**2 + sigma_diff**2)
    
    # Matern-5/2 kernel with effective length scale
    # The length scale in original space is 0.2 on [0,1], so in normalized space it's ~0.2
    # Use a reasonable effective length scale for the feature space
    l_eff = 0.3
    r = dist / l_eff
    kern = (1.0 + np.sqrt(3.0) * r) * np.exp(-np.sqrt(3.0) * r)
    
    # For each candidate, compute max correlation with other top-K candidates
    # (excluding self)
    np.fill_diagonal(kern, 0.0)
    max_corr_top = np.max(kern, axis=1)
    
    # Map this back to all candidates: candidates not in top-K get lower penalty
    # Build a penalty array for all n candidates
    penalty = np.zeros(n)
    penalty[top_k_idx] = max_corr_top
    
    # Smoothly interpolate penalty for non-top candidates based on their EI rank
    # Candidates with lower EI get less penalty (they're already less likely to be chosen)
    ei_rank = np.argsort(np.argsort(ei_norm)) / max(n - 1, 1)
    # Higher EI rank (closer to 1) means higher EI
    # Apply penalty scaled by how close the candidate is to top-K
    penalty_scale = np.clip(ei_rank * K / n, 0, 1)
    penalty = penalty * penalty_scale
    
    # Normalize penalty to [0, 1]
    pen_max = np.max(penalty)
    if pen_max > 1e-9:
        penalty_norm = penalty / pen_max
    else:
        penalty_norm = penalty
    
    # Combine: EI minus diversity penalty
    # Weight the penalty to balance exploitation and exploration
    div_weight = 0.3
    scores = ei_norm - div_weight * penalty_norm
    
    # Add a small exploration bonus for high-uncertainty regions
    # This helps escape local optima in the oscillatory Weierstrass function
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-9:
        sigma_bonus = sigma_safe / sigma_max
    else:
        sigma_bonus = np.zeros_like(sigma_safe)
    
    scores = scores + 0.1 * sigma_bonus
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
