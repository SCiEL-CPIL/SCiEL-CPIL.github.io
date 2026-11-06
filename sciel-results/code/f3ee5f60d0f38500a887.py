import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Calculate remaining capacity for each bin
    rem = bins - item
    
    # Valid bins have non-negative remaining capacity
    valid_mask = rem >= 0
    
    # Initialize scores to -inf for invalid bins
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(valid_mask):
        return scores
    
    valid_indices = np.where(valid_mask)[0]
    valid_rem = rem[valid_mask]
    
    # Strategy: Rank-based scoring with a smooth surrogate for the count objective.
    # Original: Score = -count(j != i such that bins[j] < valid_rem[i])
    # This favors bins where the resulting gap is small (Best-Fit like).
    # 
    # Improvement: Use a soft-count (log-sum-exp based) to smooth the discontinuities
    # in the rank-based score. This helps with tie-breaking and provides a more 
    # nuanced preference among bins with similar ranks.
    #
    # We compute a soft count of bins with remainder < valid_rem[i].
    # For each valid bin i, we want a smooth estimate of how many bins have remainder 
    # less than valid_rem[i]. We use a Gaussian kernel with a small bandwidth.
    
    # Get all bin remainders
    all_bins = bins
    
    # For efficiency, we'll compute the soft count using a vectorized approach.
    # soft_count[i] = sum_j exp(-((bins[j] - valid_rem[i]) / sigma)^2)
    # But this is O(n^2) which might be too slow for large n.
    #
    # Alternative: Use the rank but add a small continuous component based on the 
    # distance to the nearest neighbor in the sorted bins.
    #
    # Let's try a different approach: Instead of a hard count, use the rank but 
    # interpolate between adjacent ranks based on the position of valid_rem[i] 
    # in the sorted bins.
    
    sorted_bins = np.sort(bins)
    n = len(bins)
    
    # For each valid_rem, find the insertion position in sorted_bins
    # This gives the number of elements strictly less than valid_rem
    insert_pos = np.searchsorted(sorted_bins, valid_rem, side='left')
    
    # The hard count is insert_pos
    # To smooth, we can interpolate between insert_pos and insert_pos+1
    # based on where valid_rem falls between sorted_bins[insert_pos-1] and sorted_bins[insert_pos]
    
    # For bins where insert_pos == 0, there's no lower neighbor
    # For bins where insert_pos == n, there's no upper neighbor
    
    # Let's compute a smooth score:
    # base_score = -insert_pos (the original rank-based score)
    # smooth_correction: small adjustment based on proximity to neighbors
    
    # Actually, let's try a simpler improvement: combine the rank-based score with 
    # a small penalty for very small gaps (to avoid creating unusable fragments)
    # and a small bonus for gaps that are close to the median of current gaps.
    
    # Median of current bin remainders
    median_rem = np.median(bins)
    
    # Score = -rank + lambda * (1 - exp(-|valid_rem - median_rem| / scale))
    # This adds a preference for gaps near the median, which might be more fillable.
    
    # But the directive says to keep the design family. The rank-based score is the core.
    # Let's try: Score = -rank + small_bonus_for_common_gaps
    
    # Count how many bins have remainder close to valid_rem[i] (within a small window)
    # This is a "density" measure.
    
    # Use a histogram-like approach: for each valid_rem, count bins in [valid_rem - w, valid_rem + w]
    # where w is a small window (e.g., 1% of capacity)
    
    window = 0.01 * 100  # 1 unit window for capacity 100
    
    # Vectorized density count
    # For each valid_rem[i], count bins in [valid_rem[i] - window, valid_rem[i] + window]
    # This is O(n*m) where m is number of valid bins, which could be slow.
    
    # Alternative: Use the sorted bins and binary search
    # For each valid_rem[i], find the range in sorted_bins
    left_bounds = np.searchsorted(sorted_bins, valid_rem - window, side='left')
    right_bounds = np.searchsorted(sorted_bins, valid_rem + window, side='right')
    density_counts = right_bounds - left_bounds
    
    # Combine: Score = -rank + beta * density_counts
    # Higher density means more bins have similar gaps, which might be good for future packing.
    # But we want to minimize the rank (prefer small gaps), so -rank is the primary term.
    
    beta = 0.1  # Small weight for density
    
    valid_scores = -insert_pos.astype(float) + beta * density_counts.astype(float)
    
    scores[valid_indices] = valid_scores
    return scores
