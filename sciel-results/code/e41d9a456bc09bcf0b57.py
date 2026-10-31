"""Cap set greedy builder with geometric conflict minimization."""
import time
import zlib
import numpy as np

_start_time = None
_n_cache = None
_priority_cache = None
_deadline = 300.0
_safety_margin = 15.0

def _init(n):
    global _start_time, _n_cache, _priority_cache
    if _start_time is None:
        _start_time = time.time()
    if _n_cache is None or _n_cache != n:
        _n_cache = n
        _priority_cache = {}

def _compute_static_priority(vec, n):
    """
    Compute a static priority for a vector in {0,1,2}^n.
    The priority is based on:
    1. Coordinate balance (entropy-like measure)
    2. Penalty for vectors that are "midpoint-heavy" (many 1s)
    3. A deterministic hash-based perturbation for tie-breaking
    """
    # Count occurrences of 0, 1, 2
    n0 = 0
    n1 = 0
    n2 = 0
    for x in vec:
        if x == 0:
            n0 += 1
        elif x == 1:
            n1 += 1
        elif x == 2:
            n2 += 1
    
    target = n / 3.0
    
    # Weighted deviation: penalize 1s slightly more since they are midpoints
    # in affine lines (a, b, c are collinear iff a+b+c = 0 mod 3, and 1 is the
    # "middle" value that often appears as the average)
    deviation = (n0 - target)**2 + 1.1 * (n1 - target)**2 + (n2 - target)**2
    
    # Entropy-like balance measure
    total = n
    entropy = 0.0
    for cnt in (n0, n1, n2):
        if cnt > 0:
            p = cnt / total
            entropy -= p * np.log(p + 1e-15)
    
    # Number of distinct coordinate values
    distinct = (1 if n0 > 0 else 0) + (1 if n1 > 0 else 0) + (1 if n2 > 0 else 0)
    
    # Vectors with all same coordinate are bad
    all_same_penalty = 5.0 if distinct == 1 else 0.0
    
    # Sum of squares (distance from origin, moderate is good)
    sum_sq = 0
    for v in vec:
        sum_sq += v * v
    max_sum_sq = 4 * n
    norm_sum_sq = sum_sq / max_sum_sq if max_sum_sq > 0 else 0
    
    # Combine: high entropy is good, low deviation is good, distinct is good
    # We want to MINIMIZE deviation and MAXIMIZE entropy
    priority_val = entropy * 2.0 - deviation * 0.5 + distinct * 0.3 - all_same_penalty
    
    # Add a small deterministic perturbation based on the vector itself
    # Use CRC32 for fast, well-distributed hash
    try:
        h = zlib.crc32(bytes(vec))
    except Exception:
        h = sum(i * v for i, v in enumerate(vec)) % 1000003
    
    # Scale hash to a small perturbation that doesn't override the main heuristic
    hash_component = (h % 100000) / 100000.0 * 0.05
    
    return priority_val + hash_component

def priority(vector, n):
    """
    Returns a priority value for the given vector. Higher priority means
    the vector is considered first in the greedy insertion.
    
    Strategy: Static heuristic based on coordinate balance, midpoint penalty,
    and hash-based tie-breaking. The evaluator computes all priorities upfront
    before greedy insertion, so we use vector-intrinsic properties.
    """
    _init(n)
    
    # Deadline check
    elapsed = time.time() - _start_time
    if elapsed > _deadline - _safety_margin:
        # Return a deterministic fallback to avoid further computation
        vec = tuple(vector)
        try:
            return float(zlib.crc32(bytes(vec)))
        except Exception:
            return float(sum(vector))
    
    vec = tuple(vector)
    
    # If already cached, return cached value
    if vec in _priority_cache:
        return _priority_cache[vec]
    
    priority_val = _compute_static_priority(vec, n)
    _priority_cache[vec] = priority_val
    return priority_val
