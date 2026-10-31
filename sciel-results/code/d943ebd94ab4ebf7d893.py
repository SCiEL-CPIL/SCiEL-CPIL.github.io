"""Cap set greedy builder with geometric-aware priority."""
import time
import itertools
import numpy as np

# EVOLVE-BLOCK-START
_start_time = time.time()
_DEADLINE = 325  # seconds, safety margin before 330s kill

def _is_cap_set_safe(vec, current_set):
    """Check if adding vec to current_set keeps it a cap set.
    A cap set has no three distinct points a,b,c with a+b+c ≡ 0 mod 3.
    """
    for a in current_set:
        for b in current_set:
            if a == b or b == vec or a == vec:
                continue
            # Check if a, b, vec form an affine line: a + b + vec ≡ 0 mod 3
            # i.e., vec ≡ -a-b mod 3
            diff = [(-a[i] - b[i] - vec[i]) % 3 for i in range(len(vec))]
            if all(d == 0 for d in diff):
                return False
    return True

def priority(vector, n):
    """Priority for greedy cap set construction.
    
    Uses coordinate balance, distinct-value count, and a geometric
    penalty for vectors that are 'central' (likely to form lines).
    """
    # Deadline guard
    if time.time() - _start_time > _DEADLINE:
        return -1e18
    
    v = tuple(vector)
    n = len(v)
    
    # Count of each coordinate value
    counts = [0, 0, 0]
    for x in v:
        counts[x] += 1
    
    # Entropy of coordinate distribution
    import math
    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / n
            entropy -= p * math.log(p + 1e-15)
    
    # Number of distinct coordinate values used
    distinct = sum(1 for c in counts if c > 0)
    
    # Balance: prefer vectors with all three values present
    # and well-distributed
    balance_score = entropy + 0.3 * distinct
    
    # Geometric heuristic: vectors with coordinates closer to 1 (middle)
    # are more likely to form lines with many others. Penalize "central" vectors.
    center_deviation = sum(abs(x - 1) for x in v) / n
    # Higher center_deviation means more spread from center -> less likely to form lines
    geo_score = center_deviation
    
    # Sum of coordinates mod 3 - avoid sums that are 0 (more line-prone)
    coord_sum = sum(v) % 3
    sum_penalty = 0.1 if coord_sum == 0 else 0.0
    
    # Deterministic pseudo-random tie-breaker based on vector
    # Use a hash-like function
    h = 0
    for i, x in enumerate(v):
        h = (h * 31 + x * (i + 1)) & 0xFFFFFFFF
    tie_break = (h / 0xFFFFFFFF) * 0.01
    
    total = balance_score + 0.5 * geo_score - sum_penalty + tie_break
    return total

# EVOLVE-BLOCK-END
