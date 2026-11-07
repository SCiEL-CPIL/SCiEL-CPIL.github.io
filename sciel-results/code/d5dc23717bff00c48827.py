"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    n = len(mu)
    if n == 0:
        return np.array([])

    num_bins = 10
    # Sort indices by mu (lowest is best)
    sorted_indices = np.argsort(mu)

    # Create bin assignments
    bin_indices = np.zeros(n, dtype=int)
    bin_size = n // num_bins
    for i in range(num_bins):
        start = i * bin_size
        end = (i + 1) * bin_size
        if i == num_bins - 1:
            end = n
        # Handle case where n < num_bins
        if start >= n:
            break
        bin_indices[sorted_indices[start:end]] = i

    # Calculate statistics per bin
    max_sigma_per_bin = np.zeros(num_bins)
    mean_mu_per_bin = np.zeros(num_bins)
    
    for i in range(num_bins):
        mask = bin_indices == i
        if np.any(mask):
            max_sigma_per_bin[i] = np.max(sigma[mask])
            mean_mu_per_bin[i] = np.mean(mu[mask])
        else:
            # If a bin is empty (should not happen with this distribution, but safe)
            max_sigma_per_bin[i] = 0.0
            mean_mu_per_bin[i] = np.inf

    # Normalize mu and sigma to [0, 1]
    valid_mus = mean_mu_per_bin[mean_mu_per_bin != np.inf]
    if len(valid_mus) > 0:
        min_mu = np.min(valid_mus)
        max_mu = np.max(valid_mus)
        if max_mu > min_mu:
            norm_mu = (mean_mu_per_bin - min_mu) / (max_mu - min_mu)
        else:
            norm_mu = np.zeros(num_bins)
    else:
        norm_mu = np.zeros(num_bins)
        
    # Ensure inf mu gets norm 1 (worst)
    norm_mu[mean_mu_per_bin == np.inf] = 1.0

    valid_sigmas = max_sigma_per_bin
    if len(valid_sigmas) > 0:
        min_sigma = np.min(valid_sigmas)
        max_sigma = np.max(valid_sigmas)
        if max_sigma > min_sigma:
            norm_sigma = (max_sigma_per_bin - min_sigma) / (max_sigma - min_sigma)
        else:
            norm_sigma = np.zeros(num_bins)
    else:
        norm_sigma = np.zeros(num_bins)

    # Base score: 1.0 for best (lowest mu), 0.0 for worst
    base_scores = 1.0 - norm_mu

    # Adaptive beta based on f_best
    # f_best is the lowest observed value. Min is 0.
    # If f_best is close to 0, exploit (low beta).
    # If f_best is large, explore (high beta).
    delta = max(f_best, 0.0)
    beta = 0.5 + 1.5 * (delta / (delta + 1.0))

    bin_scores = base_scores + beta * norm_sigma

    # Assign scores to points
    scores = np.zeros(n)
    for i in range(num_bins):
        mask = bin_indices == i
        scores[mask] = bin_scores[i]

    return scores
# EVOLVE-BLOCK-END
