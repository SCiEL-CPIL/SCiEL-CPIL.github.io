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
    feasible_items = item * np.ones_like(rem)
    
    # Fragmentation-aware scoring:
    # 1. Base preference for tight fits (lower remainder = better)
    # 2. Penalize bins where the item takes 20-50% of remaining capacity
    #    (creates moderate fragments that are hard to use later)
    # 3. Small bonus for very tight fits (remainder < 5% of capacity)
    
    epsilon = 1e-6
    capacity = 100.0  # Known capacity from task description
    
    # Base score: prefer smaller remainders
    base = -rem
    
    # Fragmentation penalty: item_size / remaining_capacity ratio
    # If ratio is in [0.2, 0.5], it creates a problematic fragment
    ratio = feasible_items / (rem + epsilon)
    
    # Penalty peaks around ratio=0.35, zero outside [0.2, 0.5]
    # Using a Gaussian-like bump centered at 0.35
    center = 0.35
    width = 0.12
    frag_penalty = -2.0 * np.exp(-((ratio - center) ** 2) / (2 * width ** 2))
    # Only apply penalty when ratio is in the problematic range
    mask = (ratio > 0.2) & (ratio < 0.5)
    frag_penalty = np.where(mask, frag_penalty, 0.0)
    
    # Small bonus for very tight fits (remainder < 5% of capacity)
    tight_bonus = np.where(rem < 0.05 * capacity, 0.5, 0.0)
    
    scores[feasible] = base + frag_penalty + tight_bonus
    
    if time.time() > deadline:
        return scores
        
    return scores
