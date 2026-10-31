"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Golden ratio target for remaining capacity
# The golden ratio (~0.618) is often cited in subdivision problems as a point
# that balances fragmentation and utilization.
# Capacity is 100.
CAPACITY = 100.0
GOLDEN_RATIO = 0.6180339887498948
TARGET_REMAINING = GOLDEN_RATIO * CAPACITY  # ~61.8

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores each bin based on how close its remaining capacity is to the golden ratio target.
    The idea is to avoid both too-small remainders (fragmentation) and too-large remainders (underutilization).
    """
    # Calculate remaining space after placing the item
    remaining = bins - item
    
    # Invalidate bins that cannot fit the current item
    valid_mask = remaining >= 0
    invalid_mask = ~valid_mask
    
    # Calculate score: negative distance to target
    # We want the highest score, so we minimize the distance.
    # score = -abs(remaining - TARGET_REMAINING)
    
    # However, we also need to ensure that bins with very small remainders are penalized heavily
    # because they cannot fit future items.
    # But the absolute distance already penalizes small remainders if the target is large.
    # If remaining = 0, distance = 61.8, score = -61.8
    # If remaining = 61.8, distance = 0, score = 0
    # If remaining = 100, distance = 38.2, score = -38.2
    
    # This naturally prefers bins that leave ~61.8 space.
    
    scores = -np.abs(remaining - TARGET_REMAINING)
    
    # Invalidate bins that can't fit current item
    scores[invalid_mask] = -np.inf
    
    return scores

# EVOLVE-BLOCK-END
