"""Online bin packing: score each feasible bin to minimize mean excess over L1 lower bound."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item is placed in the bin with the highest score.
    
    Strategy: Worst Fit.
    We maximize the remaining space in the chosen bin. This preserves
    large contiguous blocks of space for future large items, which is
    particularly effective for skewed item size distributions (like Weibull)
    where a few large items can cause significant waste if they get stuck
    in fragmented bins (a common failure mode of Best Fit).
    
    For infeasible bins (insufficient capacity), we assign -inf so they are
    never selected.
    """
    remaining = bins - item
    
    # Create a score array, initialize to -inf for infeasible bins
    scores = np.full_like(remaining, -np.inf, dtype=np.float64)
    
    # For feasible bins (remaining >= 0), compute score
    feasible = remaining >= 0
    if np.any(feasible):
        # Worst Fit: Highest score goes to the bin with the most remaining space
        scores[feasible] = remaining[feasible]
    
    return scores
# EVOLVE-BLOCK-END
