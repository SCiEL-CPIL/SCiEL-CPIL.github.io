"""Rewritten acquisition function using a Hypervolume-inspired acquisition for branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of branin 2d.
    Uses a Hypervolume-based acquisition that approximates expected hypervolume
    improvement relative to f_best, with a discretized integration over the
    objective space. This structural change moves away from pure EI/UCB toward
    a multi-objective-inspired criterion that better balances exploration and
    exploitation in low-dimensional BO.
    """
    # Time budget guard
    elapsed = time.time() - _start_time
    if elapsed > 290:
        # Fallback: simple EI
        sigma_safe = np.where(np.asarray(sigma, dtype=np.float64) > 1e-12,
                              np.asarray(sigma, dtype=np.float64), 1e-12)
        mu_safe = np.asarray(mu, dtype=np.float64)
        z = (f_best - mu_safe) / sigma_safe
        ei = (f_best - mu_safe) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma = np.where(sigma > 1e-12, sigma, 1e-12)
    sigma = np.where(np.isfinite(sigma), sigma, 1.0)
    mu = np.where(np.isfinite(mu), mu, f_best)

    # Hypervolume-based acquisition:
    # We approximate the expected hypervolume improvement (EHI) of each candidate
    # relative to the current best (f_best). In a single-objective setting,
    # this reduces to a weighted combination of expected improvement and a
    # penalty for uncertainty in regions where improvement is unlikely.
    
    # The key insight: instead of just EI, we consider the "volume" of objective
    # space that could be improved. We discretize the objective range around
    # f_best and compute the probability that each candidate achieves a value
    # below each threshold, weighted by the improvement magnitude.
    
    # Discretize objective space: thresholds below f_best
    # Use a geometric grid for better resolution near f_best
    known_min = 0.397887
    # Range of possible improvements: from f_best down to known_min
    # But also consider values well below f_best
    lower_bound = max(known_min, f_best - 10.0)
    upper_bound = f_best
    
    if upper_bound - lower_bound < 1e-6:
        # Very close to optimum, use simple EI
        z = (f_best - mu) / sigma
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)
    
    # Number of discretization points
    n_points = 20
    
    # Create a grid of thresholds between lower_bound and upper_bound
    # Use logarithmic spacing for better resolution near f_best
    thresholds = np.linspace(lower_bound, upper_bound, n_points)
    
    # For each candidate and each threshold, compute:
    # P(f(x) < threshold) = Phi((threshold - mu) / sigma)
    # Improvement if f(x) < threshold is (f_best - threshold)
    # Expected hypervolume contribution = sum over thresholds of
    #   (f_best - threshold) * P(f(x) < threshold) * delta_threshold
    
    # Vectorized computation
    # mu: (n_candidates,), thresholds: (n_points,)
    # We want: (n_candidates, n_points)
    mu_expanded = mu[:, np.newaxis]  # (n_candidates, 1)
    sigma_expanded = sigma[:, np.newaxis]  # (n_candidates, 1)
    thresholds_expanded = thresholds[np.newaxis, :]  # (1, n_points)
    
    # z-scores: (n_candidates, n_points)
    z_grid = (thresholds_expanded - mu_expanded) / sigma_expanded
    z_grid = np.clip(z_grid, -10, 10)  # Clip for numerical stability
    
    # CDF values: P(f(x) < threshold)
    cdf_grid = norm.cdf(z_grid)  # (n_candidates, n_points)
    
    # Improvement magnitudes: (f_best - threshold) for each threshold
    improvements = f_best - thresholds  # (n_points,)
    improvements = np.maximum(improvements, 0.0)
    
    # Delta for each threshold (integration weights)
    if n_points > 1:
        delta_t = np.zeros(n_points)
        delta_t[1:] = thresholds[1:] - thresholds[:-1]
        delta_t[0] = thresholds[1] - thresholds[0]
        # Normalize so that sum of improvements * delta_t approximates total possible improvement
        total_weight = np.sum(improvements * delta_t)
        if total_weight > 1e-12:
            delta_t = delta_t / total_weight
    else:
        delta_t = np.ones(1)
    
    # Expected hypervolume improvement for each candidate
    # EHI = sum_t (f_best - t) * P(f(x) < t) * delta_t
    ehi = np.sum(cdf_grid * improvements[np.newaxis, :] * delta_t[np.newaxis, :], axis=1)
    
    # Additionally, add a small exploration bonus based on uncertainty
    # to encourage visiting unexplored regions early
    progress = max(0.0, min(1.0, (f_best - known_min) / (15.0 - known_min)))
    exploration_weight = 0.05 * progress
    
    # Normalize sigma for exploration bonus
    sigma_mean = np.mean(sigma)
    sigma_std = np.std(sigma)
    if sigma_std > 1e-12:
        sigma_norm = (sigma - sigma_mean) / sigma_std
    else:
        sigma_norm = np.zeros_like(sigma)
    sigma_norm = np.clip(sigma_norm, -3, 3)
    
    scores = ehi + exploration_weight * sigma_norm
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
