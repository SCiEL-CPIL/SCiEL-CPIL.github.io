"""Cost-aware acquisition for three-hump-camel 2D BO."""
import time
import numpy as np
from scipy.stats import norm

# Pre-compute a lookup table for norm.cdf and norm.pdf to speed up evaluation
# z range: EI is only significant for z in roughly [-10, 10]
_Z_LUT_MIN = -10.0
_Z_LUT_MAX = 10.0
_Z_LUT_N = 200001
_z_lut_grid = np.linspace(_Z_LUT_MIN, _Z_LUT_MAX, _Z_LUT_N)
_cdf_lut = norm.cdf(_z_lut_grid)
_pdf_lut = norm.pdf(_z_lut_grid)

def _fast_norm_cdf(z):
    """Vectorized norm.cdf using interpolation on a precomputed grid."""
    z = np.clip(z, _Z_LUT_MIN, _Z_LUT_MAX)
    # Map z to grid indices
    idx = ((z - _Z_LUT_MIN) / (_Z_LUT_MAX - _Z_LUT_MIN) * (_Z_LUT_N - 1)).astype(np.int64)
    idx = np.clip(idx, 0, _Z_LUT_N - 2)
    # Linear interpolation
    x0 = _z_lut_grid[idx]
    x1 = _z_lut_grid[idx + 1]
    y0 = _cdf_lut[idx]
    y1 = _cdf_lut[idx + 1]
    t = (z - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)

def _fast_norm_pdf(z):
    """Vectorized norm.pdf using interpolation on a precomputed grid."""
    z = np.clip(z, _Z_LUT_MIN, _Z_LUT_MAX)
    idx = ((z - _Z_LUT_MIN) / (_Z_LUT_MAX - _Z_LUT_MIN) * (_Z_LUT_N - 1)).astype(np.int64)
    idx = np.clip(idx, 0, _Z_LUT_N - 2)
    x0 = _z_lut_grid[idx]
    x1 = _z_lut_grid[idx + 1]
    y0 = _pdf_lut[idx]
    y1 = _pdf_lut[idx + 1]
    t = (z - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    with a cheapness bonus to maximize budget utilization.
    
    Returns 2000 finite scores (higher = evaluate next).
    """
    start_time = time.time()
    deadline = start_time + 250.0  # Safety margin: 250s out of 300s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma
    cdf_z = _fast_norm_cdf(z)
    pdf_z = _fast_norm_pdf(z)
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # EI per unit cost
    ei_per_cost = ei / cost
    
    # Cheapness bonus: prefer points with lower cost
    # cost is in [1, 2] range (1 + mean of coords scaled to [0,1])
    # We add a small bonus to encourage exploration of cheaper regions
    # This helps utilize the budget more efficiently
    cheapness_bonus = 0.05 * (2.0 - cost)  # Max bonus 0.05 when cost=1, 0 when cost=2
    
    # Combined score
    scores = ei_per_cost + cheapness_bonus
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Time guard
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        # If we're running out of time, return what we have
        pass
    
    return scores
