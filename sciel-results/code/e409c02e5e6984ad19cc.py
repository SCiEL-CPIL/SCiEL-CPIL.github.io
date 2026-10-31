import time
import math
from collections import Counter

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s

def priority(vector, n):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    if time.time() - _start_time > _DEADLINE:
        return 0.0
    
    # Sum of squares of coordinate values: favors 1s and 2s over 0s
    # 0^2=0, 1^2=1, 2^2=4
    # This breaks the symmetry of the standard balance heuristic and
    # exploits the non-linear distribution of affine lines in the ternary grid
    sum_sq = sum(x * x for x in vector)
    
    # Count of 1s and 2s (non-zero entries)
    c1 = vector.count(1)
    c2 = vector.count(2)
    c0 = n - c1 - c2
    
    # Number of distinct symbols
    distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    
    # Pairwise disagreement: count positions where adjacent coords differ
    # Higher disagreement = more "mixed" vector
    disagreement = 0
    for i in range(n - 1):
        if vector[i] != vector[i + 1]:
            disagreement += 1
    # Normalize by max possible (n-1)
    if n > 1:
        norm_disagree = disagreement / (n - 1)
    else:
        norm_disagree = 0.0
    
    # Hash-based tiebreaker for deterministic diversity
    # Use a simple hash of the vector to break ties
    h = 0
    for x in vector:
        h = (h * 31 + x) & 0xFFFFFFFF
    hash_frac = (h % 10000) / 10000.0 * 0.01  # small perturbation
    
    # Combine: sum_sq as primary (favors non-zero coords, breaks symmetry),
    # distinct symbols as secondary, disagreement as tertiary,
    # hash as tiny tiebreaker
    # sum_sq ranges from 0 (all 0s) to 4n (all 2s)
    # Normalize sum_sq to [0, 1]
    norm_sum_sq = sum_sq / (4.0 * n) if n > 0 else 0.0
    
    return float(10.0 * norm_sum_sq + 2.0 * distinct / 3.0 + 1.0 * norm_disagree + hash_frac)
