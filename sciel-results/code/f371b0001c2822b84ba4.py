"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(demands, capacity):
    """
    Compute an item-pair attractiveness matrix for ant colony optimization
    of offline bin packing.
    
    The heuristic guides ants to place item j after item i in a way that
    minimizes wasted space and leaves useful residuals for future items.
    """
    t0 = time.time()
    deadline = t0 + 300.0  # safety margin before 330s kill

    n = len(demands)
    d = np.array(demands, dtype=float)

    # Pair sums
    pair = d[:, None] + d[None, :]

    # Mask for pairs that fit
    fits = pair <= capacity

    # Initialize with very low attractiveness
    h = np.full((n, n), 1e-9, dtype=float)

    # Component 1: Fill ratio squared - rewards fuller packs quadratically
    fill_ratio = pair / capacity
    fill_score = np.where(fits, fill_ratio ** 2, 0.0)

    # Component 2: Inverse remaining space - rewards tight packing
    remaining = np.where(fits, capacity - pair, capacity)
    inv_remaining = np.where(remaining > 1e-6, 1.0 / (remaining + 1.0), 0.0)
    inv_remaining_score = np.where(fits, inv_remaining * capacity * 0.5, 0.0)

    # Component 3: Residual quality - smooth measure of how well leftover
    # can accommodate future items
    min_d = d.min()
    max_d = d.max()
    residual = np.where(fits, capacity - pair, 0.0)

    # Smooth residual utility: high when residual is either very small (tight pack)
    # or large enough to fit a meaningful fraction of remaining items
    # Use a smooth peak function around residual values that are "useful"
    # A residual is most useful if it's close to the median item size
    med_d = np.median(d)
    
    # Peak 1: residual close to median item size (good for next item)
    peak_res = np.exp(-((residual - med_d) ** 2) / (med_d * 0.5))
    
    # Peak 2: very tight residual (nearly full bin)
    tight_peak = np.exp(-(residual ** 2) / (min_d * 0.3))
    
    # Peak 3: large residual (can fit multiple more items)
    large_peak = np.exp(-((residual - max_d) ** 2) / (max_d * 0.8))
    
    residual_bonus = np.where(fits, 0.4 * (peak_res + tight_peak + large_peak), 0.0)

    # Combine components with adjusted weights
    h = np.where(fits, fill_score + 0.6 * inv_remaining_score + residual_bonus, 1e-9)

    # Don't allow placing an item after itself
    np.fill_diagonal(h, 1e-9)

    # Ensure non-positive and non-finite values become 1e-9
    h[~np.isfinite(h)] = 1e-9
    h[h <= 0] = 1e-9

    # Check deadline before optional enhancement
    if time.time() > deadline - 5:
        return h

    # Optional: Add a small bonus for pairs where the residual closely matches
    # some other item's size, enabling a triple fit
    # Vectorized version using np.searchsorted
    if time.time() < deadline - 10:
        sorted_d = np.sort(d)
        
        # For each pair (i,j), find the largest item that fits in residual[i,j]
        # residual[i,j] = capacity - d[i] - d[j]
        # We want to find best_fit = max(d[k]) such that d[k] <= residual[i,j]
        
        # Vectorized searchsorted: for each element in residual, find index in sorted_d
        # searchsorted returns the index where the value would be inserted
        # We want the largest value <= residual, so side='right' - 1
        idx = np.searchsorted(sorted_d, residual, side='right') - 1
        
        # Clip indices to valid range
        idx = np.clip(idx, 0, len(sorted_d) - 1)
        
        # Get the best fitting item size for each pair
        best_fit = sorted_d[idx]
        
        # Only consider pairs that fit and have positive residual
        valid_mask = fits & (residual > 0)
        
        # Closeness: how well the best item fits into the residual
        # closeness = 1 - |residual - best_fit| / residual
        with np.errstate(divide='ignore', invalid='ignore'):
            closeness = 1.0 - np.abs(residual - best_fit) / np.maximum(residual, 1e-6)
        
        closeness = np.where(valid_mask, np.maximum(closeness, 0.0), 0.0)
        
        # Bonus: reward pairs where a third item can fit well
        # Scale by how many items could potentially fit in the residual
        num_fit_estimate = residual / med_d
        fit_factor = np.where(num_fit_estimate > 1.5, 0.5, 1.0)
        bonus = closeness * 0.3 * fit_factor
        
        # Also add a complementarity bonus: reward pairs that fill ratios that
        # leave optimal residuals for the remaining items
        # For 120 items, avg ~10 items per bin, so we want fill ratios that
        # leave room for ~1-2 more items of typical size
        fill = pair / capacity
        # Typical item size is ~59.5 (mean of 20-99)
        # After placing 2 items with fill f, remaining space = (1-f)*150
        # We want remaining space to fit ~1-2 more items: 59.5 to 119
        # So (1-f)*150 in [59.5, 119] => f in [0.207, 0.603]
        # But we also want tight packing, so weight toward higher fills
        # Peak around 0.55 and 0.70
        peak1 = np.exp(-((fill - 0.55) ** 2) / 0.015)
        peak2 = np.exp(-((fill - 0.70) ** 2) / 0.012)
        complement_bonus = 0.12 * (peak1 + peak2) * fits
        
        h += bonus + complement_bonus
        np.fill_diagonal(h, 1e-9)
        h[~np.isfinite(h)] = 1e-9
        h[h <= 0] = 1e-9

    return h
# EVOLVE-BLOCK-END
