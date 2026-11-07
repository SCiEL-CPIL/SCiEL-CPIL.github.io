import numpy as np
import time
from scipy.stats import norm

# Track start time for deadline guard
_START_TIME = time.time()
_SAFETY_MARGIN = 10.0  # seconds before the 330s limit


def _time_left():
    return 330.0 - (time.time() - _START_TIME) - _SAFETY_MARGIN


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function combining Expected Improvement (EI) with
    quantile-based exploration to encourage systematic coverage of the domain.
    
    Strategy:
    1. Compute EI for exploitation
    2. Partition candidates into quantile bins by posterior mean
    3. Within each bin, reward high uncertainty (sigma) to force exploration
    4. Blend local bin scores with global EI for balanced search
    
    This prevents clustering around a single region and ensures diverse
    coverage of the search space, which is critical for the Beale function
    with its local minima.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Numerical safety
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    n = mu.shape[0]
    if n == 0:
        return np.array([], dtype=float)
    
    # Check deadline - if time is running out, return simple EI
    if _time_left() < 1.0:
        z = (f_best - mu) / sigma
        z = np.clip(z, -50, 50)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        ei = np.where(np.isfinite(ei), ei, 0.0)
        return ei
    
    # Compute Expected Improvement
    z = (f_best - mu) / sigma
    z = np.clip(z, -50, 50)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # Where mu >= f_best, EI should be ~0
    ei = np.where(mu < f_best, ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Quantile-based exploration: partition candidates into bins by posterior mean
    # Within each bin, reward high uncertainty to force diverse exploration
    order = np.argsort(mu)
    n_bins = 20
    bin_size = n // n_bins
    
    scores = np.zeros(n)
    
    for i in range(n_bins):
        bin_start = i * bin_size
        bin_end = bin_start + bin_size if i < n_bins - 1 else n
        bin_indices = order[bin_start:bin_end]
        
        if len(bin_indices) == 0:
            continue
        
        bin_mu = mu[bin_indices]
        bin_sigma = sigma[bin_indices]
        
        # Normalize sigma within the bin to [0, 1]
        sigma_min = np.min(bin_sigma)
        sigma_max = np.max(bin_sigma)
        if sigma_max - sigma_min > 1e-12:
            sigma_norm = (bin_sigma - sigma_min) / (sigma_max - sigma_min)
        else:
            sigma_norm = np.ones_like(bin_sigma)
        
        # Combine: 70% EI (normalized within bin) + 30% exploration bonus
        bin_ei = ei[bin_indices]
        ei_min = np.min(bin_ei)
        ei_max = np.max(bin_ei)
        if ei_max - ei_min > 1e-12:
            ei_norm = (bin_ei - ei_min) / (ei_max - ei_min)
        else:
            ei_norm = np.ones_like(bin_ei)
        
        bin_scores = 0.7 * ei_norm + 0.3 * sigma_norm
        scores[bin_indices] = bin_scores
    
    # Add a global EI component to ensure exploitation still matters
    global_ei_min = np.min(ei)
    global_ei_max = np.max(ei)
    if global_ei_max - global_ei_min > 1e-12:
        global_ei_norm = (ei - global_ei_min) / (global_ei_max - global_ei_min)
    else:
        global_ei_norm = np.ones_like(ei)
    
    # Final score: blend local quantile-based score with global EI
    # Use 60% local (diverse) + 40% global (exploitative)
    final_scores = 0.6 * scores + 0.4 * global_ei_norm
    
    # Adaptive exploration boost: when f_best is large, add more uncertainty weight
    # This helps early in the search when we need to explore more
    if f_best > 1.0:
        # Scale the exploration component based on how far we are from optimum
        # Beale function max is ~3000, so use a scale that transitions around f_best=50
        alpha = 0.3 * min(f_best / (f_best + 50.0), 1.0)
        # Add scaled sigma to boost exploration in uncertain regions
        sigma_norm_global = np.zeros(n)
        sigma_min_g = np.min(sigma)
        sigma_max_g = np.max(sigma)
        if sigma_max_g - sigma_min_g > 1e-12:
            sigma_norm_global = (sigma - sigma_min_g) / (sigma_max_g - sigma_min_g)
        final_scores = final_scores + alpha * sigma_norm_global
    
    # Ensure all scores are finite
    final_scores = np.nan_to_num(final_scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return final_scores
