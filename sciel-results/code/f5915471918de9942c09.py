"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 140.0  # safety margin before 150s kill
_START_TIME = time.time()


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Deadline guard: if we're too close to the limit, return a safe default
    if time.time() - _START_TIME > _DEADLINE:
        valid = bins >= item
        if not np.any(valid):
            return np.full_like(bins, -np.inf)
        # Fallback: Best Fit (minimize waste)
        scores = -(bins - item)
        scores[~valid] = -np.inf
        return scores

    # Determine which bins have enough remaining capacity
    valid = bins >= item
    
    if not np.any(valid):
        return np.full_like(bins, -np.inf)
        
    # Strategy: Polarization Heuristic
    # Goal: Encourage bins to become either "full" (small remainder) or "empty" (large remainder).
    # This reduces the number of "half-full" bins, which are difficult to pack efficiently.
    
    # Compute remainders for valid bins
    remainders = bins[valid]
    item_val = float(item)
    
    # We need to decide: should we fit this item into a bin to make it "fuller" (closer to 0 remainder)
    # or should we preserve a bin to keep it "emptier" (closer to capacity)?
    # Actually, the item MUST go into a valid bin. So we are choosing WHICH bin to use.
    # If we put the item in a bin with remainder R, the new remainder is R - item.
    # We want the new remainder to be either close to 0 (bin becomes full) or close to some large value?
    # No, the bin capacity is fixed. The "empty" state is R close to Capacity.
    # But we don't know Capacity directly from `bins` array if it's just remaining capacity.
    # Wait, the task says "bins: np.ndarray" of remaining capacity.
    # Usually, in these tasks, the array contains the *remaining* space.
    # If we assume all bins started with the same capacity C, then "empty" means R is close to C.
    # However, we don't know C. But we can estimate it or use relative measures.
    
    # Let's look at the distribution of current remainders.
    # If we pick a bin with remainder R, new remainder is R - item.
    # We want to minimize the "fragmentation".
    # A common metric for fragmentation is the number of bins with 0 < R < threshold.
    
    # Let's try a score that favors remainders that are either very small (after packing) or very large (before packing, but we are packing).
    # Actually, if we pack, the remainder DECREASES.
    # So we are choosing from {R_i - item | R_i >= item}.
    # We want the resulting set of remainders to be polarized.
    
    # Let's compute the potential new remainders.
    new_remainders = remainders - item_val
    
    # Estimate the "capacity" or upper bound of remainders.
    # The max remainder in the system is a good proxy for the "empty" state.
    max_rem = np.max(remainders)
    
    # If max_rem is 0, something is wrong, but valid ensures at least one.
    # Normalize remainders to [0, 1] relative to max_rem?
    # Or just use the distance to the extremes of the *current* population of valid bins?
    
    # Let's try: Score = - (distance of new_remainder to 0) - (distance of new_remainder to max_rem)
    # This would favor new remainders close to 0 OR close to max_rem.
    # But new_remainder is always < remainders. So it can never be close to max_rem unless max_rem is small.
    # This logic is flawed.
    
    # Alternative: Polarization of the *entire* system.
    # If we put item in bin j, the new remainder is r_j' = r_j - item.
    # The "cost" of fragmentation is often modeled as the variance of remainders or the count of "middle" remainders.
    # Let's define a "middle" zone.
    # If we have many bins, the "middle" is around the median.
    
    # Let's use a simpler, robust heuristic:
    # Prefer bins where the new remainder is either very small (good fill) or where the old remainder was very large (preserving a large bin for future large items? No, we are consuming it).
    
    # Let's look at Inspiration 2 (score 1.0204): "static, heuristic interpolation between Best Fit and Worst Fit based on global statistics".
    # Inspiration 3 (score 1.0309): uses _weibull_cdf.
    
    # The prompt suggests: "Model the bin state as a 1D spatial coordinate where the score is the negative Euclidean distance between the resulting remainder and a 'golden ratio' target".
    # But the DIRECTIVE says: "Do NOT micro-tune constants. Make a STRUCTURAL change... a different design or modelling family".
    # And the specific strategy to try in the change is: "Model the bin state as a 1D spatial coordinate where the score is the negative Euclidean distance between the resulting remainder and a 'golden ratio' target".
    
    # Wait, the DIRECTIVE explicitly tells me what strategy to try:
    # "Strategy to try in this change: Model the bin state as a 1D spatial coordinate where the score is the negative Euclidean distance between the resulting remainder and a 'golden ratio' target (approx. 0.618 * capacity), leveraging the mathematical property that the golden ratio minimizes the worst-case waste in sequential subdivision problems."
    
    # Okay, I must follow the directive's suggested strategy, even if I was thinking of polarization.
    # The directive says: "Strategy to try in this change: ..."
    # So I will implement the Golden Ratio Target strategy.
    
    # How to estimate capacity?
    # The bins array contains remaining capacities.
    # The maximum value in the bins array is likely close to the original capacity C, unless all bins have been significantly used.
    # However, if we have many bins, the max remainder might be C.
    # Let's assume C = np.max(bins) as a heuristic for the capacity.
    
    C = np.max(bins)
    if C == 0:
        C = 1.0 # Avoid division by zero or degenerate case
        
    target_remainder = 0.618 * C
    
    # For each valid bin, calculate the new remainder after placing the item.
    # new_rem = bins[i] - item
    # Score = - |new_rem - target_remainder|
    
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    # Only compute for valid bins
    valid_indices = np.where(valid)[0]
    if len(valid_indices) > 0:
        new_rems = bins[valid_indices] - item
        # Distance to the golden ratio target
        distances = np.abs(new_rems - target_remainder)
        # We want to MINIMIZE distance, so score is negative distance
        scores[valid_indices] = -distances
        
    return scores
# EVOLVE-BLOCK-END
