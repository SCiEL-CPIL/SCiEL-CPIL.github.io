"""Rewritten cost-aware acquisition for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Precompute a lookup table for the standard normal CDF and PDF to speed up
# vectorized evaluation. We use a fine grid and linear interpolation.
_Z_MIN = -10.0
_Z_MAX = 10.0
_N_GRID = 200001
_Z_GRID = np.linspace(_Z_MIN, _Z_MAX, _N_GRID)
_PDF_GRID = norm.pdf(_Z_GRID)
_CDF_GRID = norm.cdf(_Z_GRID)

def _fast_norm_cdf(z):
    """Fast vectorized CDF evaluation using interpolation."""
    z = np.clip(z, _Z_MIN, _Z_MAX)
    # Linear interpolation
    idx = (z - _Z_MIN) / (_Z_MAX - _Z_MIN) * (_N_GRID - 1)
    idx = np.clip(idx, 0, _N_GRID - 1)
    i0 = np.floor(idx).astype(np.int64)
    i1 = np.minimum(i0 + 1, _N_GRID - 1)
    frac = idx - i0
    return _CDF_GRID[i0] * (1 - frac) + _CDF_GRID[i1] * frac

def _fast_norm_pdf(z):
    """Fast vectorized PDF evaluation using interpolation."""
    z = np.clip(z, _Z_MIN, _Z_MAX)
    idx = (z - _Z_MIN) / (_Z_MAX - _Z_MIN) * (_N_GRID - 1)
    idx = np.clip(idx, 0, _N_GRID - 1)
    i0 = np.floor(idx).astype(np.int64)
    i1 = np.minimum(i0 + 1, _N_GRID - 1)
    frac = idx - i0
    return _PDF_GRID[i0] * (1 - frac) + _PDF_GRID[i1] * frac

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    with a penalty for cost, plus an exploration term based on uncertainty.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean predictions from the GP.
    sigma : ndarray, shape (2000,)
        Standard deviation predictions from the GP.
    f_best : float
        Best observed function value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate).
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: handle zero or negative sigma
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -10.0, 10.0)
    
    # Expected Improvement
    cdf_z = _fast_norm_cdf(z)
    pdf_z = _fast_norm_pdf(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Cost-aware: divide by cost to get EI per unit cost
    # Also add a small exploration bonus for high uncertainty
    # to avoid getting stuck in exploitation-only mode
    exploration_bonus = 0.1 * sigma_safe / (1.0 + sigma_safe)
    
    # Combine EI/cost with exploration bonus
    scores = ei / np.maximum(cost, 1e-8) + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
