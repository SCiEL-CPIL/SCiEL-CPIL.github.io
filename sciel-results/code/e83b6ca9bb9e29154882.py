"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(demands, capacity):
    """
    Compute an item-by-item attractiveness matrix for ACO bin packing.
    
    The heuristic rewards placing item j after item i when:
    1. The pair fits in the bin (d[i] + d[j] <= capacity).
    2. The residual capacity (capacity - d[i] - d[j]) is "good" - 
       specifically, we want residuals that can accommodate many remaining items
       but don't waste too much space.
    3. Larger items are preferred (mimicking Best Fit Decreasing).
    
    We use a residual spectrum approach: for each pair (i, j), we compute the
    residual r = capacity - d[i] - d[j]. We reward residuals that are close to
    the median of all item sizes (so they can fit typical remaining items) 
    but penalize very large residuals (wasted space) and very small/negative 
    residuals (can't fit anything else).
    """
    demands = np.asarray(demands, dtype=np.float64)
    n = len(demands)
    
    # Compute all pairwise sums
    pair_sums = demands[:, None] + demands[None, :]
    residuals = capacity - pair_sums
    
    # Only consider pairs that fit
    fits = pair_sums <= capacity
    
    # Compute the distribution of item sizes for reference
    # We want residuals that can accommodate a "typical" remaining item
    median_size = np.median(demands)
    mean_size = np.mean(demands)
    
    # Strategy: reward residuals that are in a "sweet spot"
    # - Too small (< median_size): can't fit a typical item
    # - Too large (> 2*median_size): wasting space
    # - Ideal: around median_size to 1.5*median_size
    
    # Create a smooth scoring function for residuals
    # We use a combination of:
    # 1. A penalty for residuals that are too small or too large
    # 2. A bonus for larger items (decreasing heuristic)
    
    # Base score: 1.0 for fitting pairs, 1e-9 for non-fitting
    base = np.where(fits, 1.0, 1e-9)
    
    # Residual quality score: Gaussian-like peak around ideal residual
    # Ideal residual is around median_size (can fit a typical item)
    # We use a soft-max style scoring
    
    # Compute residual quality: higher when residual is in [0.5*median, 1.5*median]
    # Use a smooth function: exp(-((residual - ideal_residual) / sigma)^2)
    ideal_residual = median_size
    sigma = median_size * 0.5
    
    # Only compute for fitting pairs
    residual_quality = np.zeros_like(residuals)
    mask = fits
    if np.any(mask):
        r = residuals[mask]
        # Clamp to avoid extreme values
        r_clamped = np.clip(r, 0, 3 * median_size)
        residual_quality[mask] = np.exp(-((r_clamped - ideal_residual) ** 2) / (2 * sigma ** 2))
    
    # Item size bonus: prefer larger items (decreasing heuristic)
    # Scale the bonus so it's meaningful
    size_bonus = demands[None, :] / (np.max(demands) + 1e-9)
    
    # Combine: base * (residual_quality + size_bonus_weight * size_bonus)
    # Weight the size bonus to encourage larger items earlier
    size_bonus_weight = 0.3
    
    heuristic = base * (residual_quality + size_bonus_weight * size_bonus)
    
    # Ensure non-fitting pairs get 1e-9
    heuristic = np.where(fits, heuristic, 1e-9)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
