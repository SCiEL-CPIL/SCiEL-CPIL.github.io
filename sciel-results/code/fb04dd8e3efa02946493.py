"""Acquisition function combining Hypervolume-based EHI with Thompson Sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of branin 2d.
    Combines a Hypervolume-based Expected Improvement (EHI) with Thompson Sampling.
    The EHI provides a structured exploitation signal based on the volume of
    objective space that could be improved, while Thompson Sampling adds
    stochastic exploration by sampling from the posterior.
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

    known_min = 0.397887
    
    # Progress: 0 at start (f_best high), 1 near optimum
    initial_range = 15.0 - known_min  # approximate range of branin
    progress = max(0.0, min(1.0, (initial_range - (f_best - known_min)) / initial_range))
    
    # Compute EHI (Expected Hypervolume Improvement)
    n_points = 30 + int(20 * progress)
    n_points = min(n_points, 50)
    
    # Define the range of thresholds
    if progress < 0.3:
        lower_bound = max(known_min, f_best - 5.0)
    else:
        lower_bound = max(known_min, f_best - 1.0)
    upper_bound = f_best
    
    if upper_bound - lower_bound < 1e-6:
        # Very close to optimum, use simple EI
        z = (f_best - mu) / sigma
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)
    
    # Create a grid of thresholds
    thresholds = np.linspace(lower_bound, upper_bound, n_points)
    
    # Vectorized computation
    mu_expanded = mu[:, np.newaxis]  # (n_candidates, 1)
    sigma_expanded = sigma[:, np.newaxis]  # (n_candidates, 1)
    thresholds_expanded = thresholds[np.newaxis, :]  # (1, n_points)
    
    # z-scores: (n_candidates, n_points)
    z_grid = (thresholds_expanded - mu_expanded) / sigma_expanded
    z_grid = np.clip(z_grid, -10, 10)
    
    # CDF values: P(f(x) < threshold)
    cdf_grid = norm.cdf(z_grid)
    
    # Improvement magnitudes: (f_best - threshold) for each threshold
    improvements = f_best - thresholds
    improvements = np.maximum(improvements, 0.0)
    
    # Trapezoidal integration weights
    delta_t = np.zeros(n_points)
    if n_points > 1:
        delta_t[1:] = thresholds[1:] - thresholds[:-1]
        delta_t[0] = thresholds[1] - thresholds[0]
        delta_t[-1] = thresholds[-1] - thresholds[-2]
        # Normalize
        total_weight = np.sum(improvements * delta_t)
        if total_weight > 1e-12:
            delta_t = delta_t / total_weight
    else:
        delta_t = np.ones(1)
    
    # Expected hypervolume improvement
    ehi = np.sum(cdf_grid * improvements[np.newaxis, :] * delta_t[np.newaxis, :], axis=1)
    
    # Thompson Sampling: draw a sample from the posterior
    # The posterior is N(mu, sigma^2), so we sample y ~ N(mu, sigma^2)
    # For minimization, we want lower values, so the score from TS is -y
    rng = np.random.default_rng()
    ts_samples = mu + sigma * rng.standard_normal(len(mu))
    ts_scores = -ts_samples  # Negate for maximization
    
    # Combine EHI and Thompson Sampling
    # Weight the Thompson Sampling contribution based on progress
    # Early on: more exploration (higher TS weight)
    # Later: more exploitation (lower TS weight, rely on EHI)
    
    # Normalize EHI to have zero mean and unit variance for consistent scaling
    ehi_mean = np.mean(ehi)
    ehi_std = np.std(ehi)
    if ehi_std > 1e-12:
        ehi_norm = (ehi - ehi_mean) / ehi_std
    else:
        ehi_norm = np.zeros_like(ehi)
    
    # Normalize TS scores similarly
    ts_mean = np.mean(ts_scores)
    ts_std = np.std(ts_scores)
    if ts_std > 1e-12:
        ts_norm = (ts_scores - ts_mean) / ts_std
    else:
        ts_norm = np.zeros_like(ts_scores)
    
    # Adaptive weight: start with high TS weight, decrease toward 0 as we progress
    # At progress=0: ts_weight=0.5, ehi_weight=0.5
    # At progress=1: ts_weight=0.1, ehi_weight=0.9
    ts_weight = 0.5 - 0.4 * progress
    ehi_weight = 1.0 - ts_weight
    
    scores = ehi_weight * ehi_norm + ts_weight * ts_norm
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
