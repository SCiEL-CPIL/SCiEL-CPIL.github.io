import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
DEADLINE = 148.0  # 150s limit, 2s safety margin
SAFETY_MARGIN = 2.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Deadline check
    if time.time() > DEADLINE - SAFETY_MARGIN:
        # Fallback to simple Best Fit
        feasible = bins >= item
        scores = np.full_like(bins, -np.inf, dtype=float)
        if np.any(feasible):
            scores[feasible] = -(bins[feasible] - item)
        return scores

    C = 100.0
    n_bins = len(bins)
    
    # Identify feasible bins
    feasible = bins >= item
    if not np.any(feasible):
        return np.full_like(bins, -np.inf, dtype=float)
    
    scores = np.full_like(bins, -np.inf, dtype=float)
    b = bins[feasible]
    
    # Strategy: Pure Best Fit (Seed)
    # The seed program (score 1.0000) uses Best Fit.
    # The previous attempt (hybrid BF/WF) scored 0.9998, indicating that deviating
    # from Best Fit for large items was detrimental for this specific Weibull distribution.
    # Best Fit minimizes the remaining space, which tends to consolidate items
    # and reduce the number of open bins, which is generally superior for
    # minimizing the total number of bins used (L1 lower bound excess).
    
    # score = -remaining_space
    # Maximizing this score minimizes the remaining space.
    scores[feasible] = -(b - item)
    
    return scores
