"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity for the item.
    The item goes to the bin with the highest score.
    
    Strategy: Best-fit with a small penalty for creating near-empty bins.
    We prefer bins where the remaining capacity after placing the item is small,
    but we add a small linear penalty to avoid pathological cases where a bin
    becomes almost empty (which wastes capacity for future items).
    """
    # Feasibility: only bins with enough remaining capacity
    remaining = bins - item
    feasible = remaining >= 0
    
    # If no feasible bins, return all -inf (should not happen per problem setup)
    if not np.any(feasible):
        return np.full_like(bins, -np.inf, dtype=float)
    
    # For feasible bins, compute the waste (remaining capacity after placing item)
    waste = np.where(feasible, remaining, 1e9)
    
    # Score = -waste + small bonus for having more total capacity used
    # The bonus encourages filling bins more completely
    # Use a small coefficient to not overwhelm the best-fit objective
    # Capacity is 100, so remaining is in [0, 100]
    
    # Primary objective: minimize waste (best-fit)
    # Secondary objective: prefer bins with more used capacity (to fill bins)
    used = np.where(feasible, bins - remaining, 0.0)
    
    # Score: higher is better
    # -waste favors tighter fits
    # +0.1 * used favors bins that are already partially filled
    score = -waste + 0.1 * used
    
    # For infeasible bins, set score to -inf
    score = np.where(feasible, score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
