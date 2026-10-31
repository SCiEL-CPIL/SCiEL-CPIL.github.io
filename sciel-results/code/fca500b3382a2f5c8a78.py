import numpy as np
import time

_START_TIME = time.time()
DEADLINE = 150.0
SAFETY_MARGIN = 2.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Deadline check
    if time.time() > DEADLINE - SAFETY_MARGIN:
        # Fallback to simple Best Fit
        feasible = bins >= item
        scores = np.full_like(bins, -np.inf, dtype=float)
        if np.any(feasible):
            rem = bins[feasible] - item
            scores[feasible] = -rem
        return scores

    C = 100.0
    n_bins = len(bins)
    
    # Identify feasible bins
    feasible = bins >= item
    if not np.any(feasible):
        return np.full_like(bins, -np.inf, dtype=float)
    
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    rem = bins[feasible] - item
    load = bins[feasible]
    
    # Hybrid Strategy based on item size relative to capacity
    # For large items (> C/2), use Best Fit to preserve larger bins for other large items.
    # For small items (<= C/2), use a "Fluid" / Ratio heuristic to minimize relative waste.
    
    if item > 50.0:
        # Best Fit: Minimize remaining space
        scores[feasible] = -rem
    else:
        # Fluid Flow Heuristic:
        # Prioritize bins where the item constitutes a large fraction of the remaining capacity.
        # This minimizes the relative "waste" of opening new bins or leaving small gaps.
        # Score is proportional to (item / remaining_capacity).
        # To avoid division by zero (if rem is 0), we add a small epsilon or handle it.
        # If rem is 0, it's a perfect fit, which is ideal.
        
        # We want to maximize item/rem.
        # So score = item / (rem + epsilon)
        # But we also want to consider the current load to consolidate?
        # The directive says: "prioritizing bins where the item constitutes a large fraction of the remaining capacity"
        
        eps = 1e-6
        ratio = item / (rem + eps)
        
        # We can also incorporate a penalty for very small remainders that are not zero?
        # The knowledge from the run said: "Penalizing very small remainders (<5) in Best Fit... improved island 4"
        # But that was for Best Fit. Here we are maximizing the ratio.
        # If rem is small, ratio is large. So we are naturally preferring small remainders.
        # However, if rem is too small (e.g., 0.1), it might be unusable for future items.
        # But since we are maximizing ratio, we prefer rem=0 (perfect fit) > rem=0.1 > rem=5.
        # This seems consistent with Best Fit but with a non-linear scaling.
        
        # Let's try a combination:
        # score = ratio - penalty_if_rem_too_small_but_not_zero?
        # Or just stick to the ratio as per the directive.
        
        scores[feasible] = ratio
        
        # Optional: Add a small bonus for bins that are already "fuller" to encourage consolidation?
        # The directive mentions "fluid flow" and "ratio of current load to item size" in the text but then specifies
        # "prioritizing bins where the item constitutes a large fraction of the remaining capacity".
        # I will stick to the specific instruction: item / remaining_capacity.
        
        # To make it more robust, we can normalize or adjust.
        # But raw ratio should work.
        
        # One issue: if item is very small, ratio is small.
        # If item is large (but <= 50), ratio is larger for same rem.
        # This seems reasonable.
        
        # Let's add a small tie-breaker: prefer lower indices if scores are very close?
        # NumPy argmax will pick the first one if there are ties.
        # Since scores are floats, ties are unlikely unless rem is identical.
        
        # To be safe against floating point precision issues causing "ties" where we might want a specific behavior:
        # We can subtract a tiny epsilon * index.
        indices = np.arange(n_bins)
        scores[feasible] -= 1e-9 * indices[feasible]

    return scores
