"""Online bin packing: score each feasible bin to minimize mean excess over L1 lower bound."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item is placed in the bin with the highest score.
    
    Strategy: Worst Fit.
    This approach places the item in the bin with the maximum remaining capacity.
    This is distinct from Best Fit (minimizing remaining space).
    Worst Fit is often effective for distributions with variable item sizes (like Weibull)
    as it preserves large empty spaces for potentially large future items,
    reducing the chance of needing to open new bins for outliers.
    """
    # bins represents the remaining capacity of each bin.
    # Feasible bins are those where bins[i] >= item.
    
    # Initialize scores to -inf to exclude infeasible bins
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    
    feasible = bins >= item
    if np.any(feasible):
        # Worst Fit: Maximize the remaining capacity.
        # We simply return the bin's remaining capacity as the score.
        # Higher score -> larger remaining capacity -> selected.
        scores[feasible] = bins[feasible]
    
    return scores
# EVOLVE-BLOCK-END
