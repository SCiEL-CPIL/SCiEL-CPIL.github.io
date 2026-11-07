"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Quantile-binned exploration acquisition function.
    
    Divides candidates into quantile bins based on posterior mean (mu).
    Within each bin, rewards high uncertainty (sigma) to encourage
    systematic coverage of diverse posterior-mean regions, preventing
    clustering in local basins on Beale's narrow valley.
    """
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Number of quantile bins
    n_bins = max(1, int(np.sqrt(n)))  # e.g., ~45 bins for 2000 candidates
    
    # Compute quantile bins based on posterior mean
    # We want to explore across the range of mu values
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    
    if mu_max - mu_min < 1e-12:
        # All mu values are essentially the same; fall back to pure uncertainty
        return sigma_safe.copy()
    
    # Create quantile bins: each bin gets an equal number of candidates
    # Sort candidates by mu and assign bin indices
    sorted_idx = np.argsort(mu)
    bin_indices = np.zeros(n, dtype=np.int64)
    
    # Assign equal-sized bins
    bin_size = n // n_bins
    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size if i < n_bins - 1 else n
        bin_indices[sorted_idx[start:end]] = i
    
    # Within each bin, compute a score that balances exploitation and exploration
    # Score = alpha * (f_best - mu) + beta * sigma
    # But we want to encourage exploration within each bin, so weight sigma more
    
    # Compute per-bin statistics
    scores = np.zeros(n)
    
    for b in range(n_bins):
        mask = (bin_indices == b)
        if not np.any(mask):
            continue
        
        mu_b = mu[mask]
        sigma_b = sigma_safe[mask]
        
        # In each bin, reward high sigma (exploration)
        # Also add a small exploitation term
        # Normalize within bin to make it scale-invariant
        mu_mean_b = np.mean(mu_b)
        sigma_mean_b = np.mean(sigma_b)
        
        # Score: higher sigma is better within the bin,
        # but also prefer lower mu (closer to best)
        # Use a weighted combination
        # Exploration weight: emphasize sigma
        # Exploitation weight: emphasize (f_best - mu)
        
        # To make it more robust, normalize both terms
        mu_range = np.max(mu_b) - np.min(mu_b) + 1e-12
        sigma_range = np.max(sigma_b) - np.min(sigma_b) + 1e-12
        
        # Normalized exploitation: (f_best - mu) normalized
        # Higher means closer to best
        exp_term = (f_best - mu_b) / (abs(f_best - mu_mean_b) + 1e-12)
        
        # Normalized exploration: sigma normalized
        exp_term = np.clip(exp_term, -10, 10)
        expl_term = sigma_b / (sigma_mean_b + 1e-12)
        
        # Weight: 70% exploration, 30% exploitation within each bin
        # This encourages diverse exploration
        bin_scores = 0.7 * expl_term + 0.3 * exp_term
        
        scores[mask] = bin_scores
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
