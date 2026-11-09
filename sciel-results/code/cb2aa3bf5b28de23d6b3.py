import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    t0 = time.time()
    deadline = t0 + 115.0
    
    remaining = bins - item
    feasible = remaining >= 0
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(feasible):
        return scores
    
    rem = remaining[feasible]
    
    # Non-linear scoring with reduced aggressiveness
    # Using fourth root instead of square root to be less aggressive
    # toward tight fits, allowing more flexibility for future items
    epsilon = 1e-6
    base = -(rem + epsilon) ** 0.25
    
    scores[feasible] = base
    
    if time.time() > deadline:
        return scores
        
    return scores
