"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Adaptive Bin Packing Heuristic.
    
    Combines Best-Fit and Worst-Fit strategies based on item size.
    - Small items: Best-Fit (minimize remaining space) to pack densely.
    - Large items: Worst-Fit (maximize remaining space) to ensure space for future large items.
    - Medium items: Slight Best-Fit bias.
    
    Score is designed such that higher score is preferred.
    """
    C = 100.0
    
    # Calculate remaining capacity
    rem = bins - item
    
    # Initialize scores to -inf for invalid bins
    scores = np.full_like(rem, -np.inf, dtype=float)
    
    valid = rem >= 0
    
    if np.any(valid):
        # Determine adaptive weight
        # ratio = item / Capacity
        # We want to transition from Best-Fit (weight < 0) to Worst-Fit (weight > 0)
        # Best-Fit: Score ~ -rem  => weight = -1
        # Worst-Fit: Score ~ rem  => weight = 1
        
        ratio = item / C
        # Clip to [0, 1] for safety, though valid bins imply item <= bin <= C (mostly)
        # Note: bins can be less than C if they are partially filled? 
        # The problem states "bins with enough remaining capacity". 
        # The `bins` array passed in likely represents the REMAINING capacity of each bin?
        # Let's re-read the contract carefully.
        # "bins: np.ndarray of one score per bin with enough remaining capacity"
        # Wait, the input is `bins`. 
        # "priority(item: float, bins: np.ndarray)"
        # "scores each bin with enough remaining capacity"
        # Usually in these interfaces, `bins` represents the *remaining* capacity of each bin.
        # Let's check the seed program:
        # `return -(bins - item)`
        # If `bins` was total capacity, `bins - item` would be remaining.
        # If `bins` was remaining capacity, `bins - item` would be remaining AFTER placement.
        # The seed program is Best-Fit. Best-Fit minimizes remaining space AFTER placement.
        # So `bins` must be the **remaining capacity** of each bin.
        # Because if `bins` was total capacity, we wouldn't know how much is already used.
        # So `rem = bins - item` is the new remaining capacity.
        
        # Okay, so `bins` is the array of current remaining capacities.
        
        ratio = item / C
        ratio = np.clip(ratio, 0.0, 1.0)
        
        # Weight mapping:
        # item=0 -> w=-1.1 (Strong Best Fit)
        # item=C -> w=0.9 (Moderate Worst Fit)
        w = 2.0 * ratio - 1.1
        
        # Apply weight to remaining capacity
        # If w < 0, higher score corresponds to lower rem (Best Fit).
        # If w > 0, higher score corresponds to higher rem (Worst Fit).
        scores[valid] = w * rem[valid]
        
    return scores
# EVOLVE-BLOCK-END
