"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity using a Worst-Fit heuristic.
    The item goes to the highest-scoring bin.
    
    Worst-Fit places items into the bin with the most remaining capacity after
    placement, preserving smaller bins for future small items and reducing
    fragmentation.
    
    Bins that cannot fit the item get -inf so they are never selected.
    """
    remaining = bins - item
    # Prefer bins with the most remaining capacity (worst-fit)
    # Higher remaining capacity = higher score
    score = np.where(remaining >= 0, remaining, -np.inf)
    return score
# EVOLVE-BLOCK-END
