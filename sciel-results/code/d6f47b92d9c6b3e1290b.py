"""Cap set greedy construction with refined spatial-spread priority heuristic."""
import time
import math
import random
from itertools import product


def priority(vector, n):
    """
    Static priority function for cap set greedy construction.
    
    Primary: Balance of digit counts (0s, 1s, 2s) - lower variance = higher priority.
    Secondary: Spatial spread of 1-positions - more spread = higher priority.
    Tertiary: Position-weighted sum to break ties structurally.
    Quaternary: Raw sum for final tie-breaking.
    """
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    target = n / 3.0
    penalty = (c0 - target) ** 2 + (c1 - target) ** 2 + (c2 - target) ** 2
    
    # Spatial spread of 1-positions: measure how spread out the 1s are
    # More spread = less likely to form small APs as midpoint
    pos_ones = [i for i in range(n) if vector[i] == 1]
    if len(pos_ones) >= 2:
        mean_pos = sum(pos_ones) / len(pos_ones)
        spread = sum((p - mean_pos) ** 2 for p in pos_ones)
    elif len(pos_ones) == 1:
        spread = 0.0
    else:
        spread = 0.0
    
    # Normalize spread to [0, 1] range for consistent scaling
    # Max possible spread for k ones in n positions is roughly n^2/12 * k
    max_spread = n * n / 12.0 if n > 0 else 1.0
    norm_spread = min(spread / max_spread, 1.0)
    
    # Position-weighted sum: encourages spreading non-zero digits across positions
    pos_weighted = sum(vector[i] * (i + 1) for i in range(n))
    
    # Raw sum
    s = sum(vector)
    
    # Combine with clear hierarchy:
    # penalty is in [0, n], norm_spread in [0,1], pos_weighted in [0, n*(n+1)]
    # Scale: penalty dominates, then spread, then pos_weighted, then sum
    return -penalty * 10000.0 + norm_spread * 500.0 + pos_weighted * 1.0 + s * 0.1
