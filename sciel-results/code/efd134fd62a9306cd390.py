"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores bins using a Harmonic-inspired heuristic.
    - Identifies the 'class' of the item based on its size relative to capacity.
    - Prefers bins whose remaining capacity falls within the range typical for that class.
    - Among suitable bins, prefers the one with the least remaining capacity (Best-Fit).
    - If no suitable bin exists, returns -inf for all bins, implying a new bin should be opened.
    """
    C = 100.0
    
    # Assume 'bins' contains only bins with enough capacity (as per seed behavior analysis)
    # But to be robust, we filter just in case.
    valid_mask = bins >= item
    
    if not np.any(valid_mask):
        return np.full_like(bins, -np.inf)
    
    # Calculate Harmonic class k
    # k = ceil(log2(C / item))
    # If item is very small, k is large.
    # If item > C, no fit (handled by valid_mask)
    
    if item <= 0:
        # Should not happen, but handle gracefully
        return np.zeros_like(bins)
        
    ratio = C / item
    # Ensure ratio >= 1
    if ratio < 1.0:
        return np.full_like(bins, -np.inf)
        
    k = np.ceil(np.log2(ratio))
    
    # Define the target range for remaining capacity
    # Bins in class k are expected to have remaining capacity in [C/2^k, C/2^(k-1))
    lower_bound = C / (2 ** k)
    upper_bound = C / (2 ** (k - 1))
    
    # Identify bins that are valid AND fall into the target range
    # Note: bins are remaining capacities.
    target_mask = valid_mask & (bins >= lower_bound) & (bins < upper_bound)
    
    scores = np.full_like(bins, -np.inf)
    
    if np.any(target_mask):
        # Among target bins, use Best-Fit scoring: minimize remaining capacity
        # Score = -(bins - item)
        # Higher score (closer to 0) is preferred.
        scores[target_mask] = -(bins[target_mask] - item)
    else:
        # No bin in the target class.
        # Harmonic algorithm would open a new bin.
        # We signal this by returning -inf for all bins.
        # If the evaluator does not handle -inf by opening a new bin,
        # this might degrade performance, but it's the correct algorithmic choice.
        pass
        
    return scores
# EVOLVE-BLOCK-END
