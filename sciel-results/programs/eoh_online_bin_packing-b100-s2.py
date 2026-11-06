"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import math

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Higher score = better choice.
    
    Uses a Harmonic-like heuristic:
    1. Determine the 'level' of the item based on its size.
    2. Determine the 'level' of each bin based on its used capacity.
    3. Prefer bins whose level matches the item's level.
    4. Among matching levels, prefer the tightest fit (Best Fit).
    5. If no matching level, prefer lower levels (larger bins/less used) as fallback?
       Actually, Harmonic falls back to lower levels (larger capacity bins).
    """
    # bins contains remaining capacities
    # Total capacity is 100.0
    C = 100.0
    
    # Calculate used capacity for each bin
    used = C - bins
    
    # Calculate item level
    # k_item such that C / 2^(k_item+1) < item <= C / 2^k_item
    # 2^k_item <= C / item
    # k_item = floor(log2(C / item))
    if item <= 0:
        return np.zeros(len(bins)) # Should not happen
        
    k_item = int(math.floor(math.log2(C / item)))
    
    # Calculate bin levels
    # For used == 0, level is very high (empty bin)
    # For used > 0, level = floor(log2(C / used))
    # Note: If used is very small, log is large.
    # If used is close to C, log is small.
    
    # Handle used=0
    safe_used = np.where(used > 0, used, 1e-9)
    k_bins = np.floor(np.log2(C / safe_used)).astype(int)
    
    # Valid bins: bins >= item
    valid = bins >= item
    
    if not np.any(valid):
        return np.full(len(bins), -np.inf)
        
    scores = np.full(len(bins), -np.inf)
    
    # For valid bins, calculate score
    # Primary: Match level. Prefer k_bins == k_item.
    # Secondary: Tightest fit (minimize remaining).
    
    # Penalty for level mismatch
    # We want k_bins to be close to k_item.
    # Harmonic algorithm: If item is in class k, place in class k.
    # If class k is full, place in class k-1 (larger bins).
    # So we prefer k_bins == k_item, then k_bins == k_item - 1, etc.
    # Actually, if we place item in a bin of class k_bin, the new used is used + item.
    # The new level might change.
    
    # Simple heuristic:
    # Score = - (abs(k_bins - k_item) * 1000 + (bins - item))
    # This strongly prefers level match, then tightest fit.
    
    level_diff = np.abs(k_bins[valid] - k_item)
    waste = bins[valid] - item
    
    # Combine
    # If level matches, diff is 0.
    # If not, diff is > 0.
    # We want to minimize level_diff first, then waste.
    
    scores_valid = - (level_diff * 1000.0 + waste)
    
    scores[valid] = scores_valid
    
    return scores
# EVOLVE-BLOCK-END
