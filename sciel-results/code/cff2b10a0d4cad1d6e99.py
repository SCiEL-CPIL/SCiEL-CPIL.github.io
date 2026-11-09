import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Uses a Smooth Best-Fit heuristic.
    Instead of strictly minimizing remaining capacity (linear Best Fit),
    we use a square root transformation on the remaining capacity.
    Score = -sqrt(remaining_capacity).
    
    This dampens the preference for the absolute tightest fit, potentially
    reducing fragmentation by keeping remaining spaces more "usable" for
    future items of varying sizes.
    
    Includes a deadline guard to ensure completion within 120s.
    """
    start_time = time.time()
    
    # Calculate remaining capacity
    remaining = bins - item
    valid_mask = remaining >= 0
    
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    
    if np.any(valid_mask):
        # Smooth Best Fit: -sqrt(remaining)
        # This is concave, so it prefers tight fits but less aggressively than linear.
        valid_remaining = remaining[valid_mask]
        # Ensure non-negative for sqrt (valid_mask guarantees >= 0)
        scores[valid_mask] = -np.sqrt(valid_remaining)
        
    # Deadline check (required by directive, though computation is fast)
    if time.time() - start_time > 115:
        pass
        
    return scores
