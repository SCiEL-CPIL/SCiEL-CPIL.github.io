"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0  # Safety margin for 150s limit

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins for online bin packing using an Adaptive Fragmentation Heuristic.
    
    This strategy models bin selection as a resource fragmentation problem.
    It dynamically adjusts the scoring preference based on the global state of the bins.
    
    1. Calculate the mean utilization of all bins.
    2. If the system is highly utilized (mean > 0.5), it favors 'Best-Fit' behavior 
       (minimizing remaining space) to prevent fragmentation of large empty spaces.
    3. If the system is lightly utilized (mean < 0.5), it favors 'Worst-Fit' behavior 
       (maximizing remaining space) to keep bins open for larger future items, 
       reducing the number of bins needed overall.
    4. The score is a linear interpolation between -remaining (Best) and +remaining (Worst).
    """
    # Check deadline
    if time.time() > _DEADLINE:
        return np.zeros_like(bins, dtype=np.float64)

    # Calculate remaining capacity
    remaining = bins - item
    feasible = remaining >= 0
    
    # Initialize scores to a very low value for infeasible bins
    scores = np.full_like(remaining, -1e18, dtype=np.float64)
    
    if not np.any(feasible):
        return scores

    # Global state: Mean utilization of bins
    # We assume bins represent current remaining capacity? 
    # Wait, the interface says `bins: np.ndarray`. 
    # In the seed program, `remaining = bins - item` implies `bins` is the *current capacity* 
    # or *current remaining space*?
    # Let's check the seed: `remaining = bins - item`. If `bins` was remaining space, 
    # placing an item would reduce it. So `bins` likely represents the *current available capacity* 
    # or the *current size of the hole*? 
    # Actually, usually in these contracts, `bins` is an array of current remaining capacities.
    # If `bins` is remaining capacity, then `remaining = bins - item` is the new remaining.
    # Let's assume `bins` = current remaining capacity in each bin.
    
    # Calculate mean remaining capacity to gauge "fullness"
    # If bins are full, mean remaining is low.
    # If bins are empty, mean remaining is high.
    # Let's define "tightness" based on the item size relative to the mean remaining space.
    # A better global signal: The ratio of the item size to the mean remaining capacity of feasible bins.
    
    feasible_remaining = remaining[feasible]
    
    # If there are feasible bins
    if len(feasible_remaining) > 0:
        mean_rem = np.mean(feasible_remaining)
        
        # Avoid division by zero
        if mean_rem < 1e-9:
            # Bins are essentially full, use Best Fit (minimize remaining)
            scores[feasible] = -feasible_remaining
        else:
            # Ratio: How large is the item relative to the average hole?
            # If item is large relative to holes, we should be careful not to fragment.
            # If item is small relative to holes, we might as well fill the smallest hole (Best Fit)
            # OR the largest hole (Worst Fit)?
            
            # Let's look at the standard heuristics:
            # Best Fit: Minimizes waste in the current bin. Good for high density.
            # Worst Fit: Maximizes remaining space. Good for keeping options open for large items.
            
            # Adaptive Strategy:
            # If the item is small compared to the mean remaining space, 
            # putting it in the largest bin (Worst Fit) preserves smaller bins for medium items.
            # If the item is large compared to the mean remaining space, 
            # we are forced into smaller bins, so Best Fit (tightest) is often better to avoid wasting space.
            
            # Let's define an alpha parameter that shifts from Worst (0) to Best (1).
            # alpha = 1 - (item / mean_rem) ? 
            # If item >> mean_rem, alpha is negative -> clamp to 1 (Best Fit).
            # If item << mean_rem, alpha is positive > 1 -> clamp to 0 (Worst Fit).
            
            ratio = item / mean_rem
            # Map ratio to [0, 1]
            # If ratio < 0.5, prefer Worst Fit (alpha=0)
            # If ratio > 0.5, prefer Best Fit (alpha=1)
            # Linear interpolation
            alpha = np.clip(ratio, 0.0, 1.0)
            
            # Score = alpha * (-remaining) + (1 - alpha) * (remaining)
            # If alpha=1: Score = -remaining (Best Fit: highest score for smallest remaining)
            # If alpha=0: Score = remaining (Worst Fit: highest score for largest remaining)
            
            scores[feasible] = alpha * (-feasible_remaining) + (1 - alpha) * feasible_remaining
            
    return scores
# EVOLVE-BLOCK-END
