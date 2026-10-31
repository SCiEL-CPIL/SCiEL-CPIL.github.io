import time
import math

_start_time = None
_DEADLINE = 300 * 0.92  # 276s safety margin

# Cache for fast hash-based tie-breaking
_hash_cache = {}

def _fast_hash(vector, n):
    """Fast deterministic hash for tie-breaking."""
    key = vector
    if key in _hash_cache:
        return _hash_cache[key]
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i * 7) & 0x7FFFFFFF
    _hash_cache[key] = h / 0x7FFFFFFF
    return _hash_cache[key]

def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.
    
    Strategy: Smooth surrogate heuristic with danger-aware scoring.
    
    1. Primary: Smooth quadratic distinct-symbol score (strongly prefer all 3 symbols)
    2. Secondary: Smooth variance-based danger proxy (penalize extreme vectors)
    3. Tertiary: Quadratic balance penalty (prefer balanced symbol distribution)
    4. Quaternary: Symmetry breaking (favors 2s to avoid sub-cube clustering)
    5. Quinary: Fast hash for deterministic tie-breaking
    
    All O(n) per call with minimal overhead.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        return 0.0

    # Fast single-pass statistics
    c0 = 0
    c1 = 0
    c2 = 0
    s = 0
    s2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
            s += 1
            s2 += 1
        else:
            c2 += 1
            s += 2
            s2 += 4

    # Factor 1: Smooth quadratic distinct-symbol score
    # 0 distinct -> 0, 1 -> 10000, 2 -> 40000, 3 -> 90000
    num_distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    distinct_score = 10000.0 * num_distinct * num_distinct

    # Factor 2: Smooth variance-based danger proxy
    # High variance = more extreme coordinates = more likely to form 3-APs
    mean = s / n
    variance = s2 / n - mean * mean
    # Scale variance to [0, 4/n] range, penalize with smooth quadratic
    danger_score = -variance * 5000.0

    # Factor 3: Quadratic balance penalty (L2 deviation from uniform)
    ideal = n / 3.0
    dev_sq = (c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2
    balance_score = -dev_sq * 5.0

    # Factor 4: Symmetry breaking - favor vectors with more 2s
    weight_score = c2 * 100.0

    # Factor 5: Fast hash for tie-breaking
    hash_score = _fast_hash(vector, n)

    # Combined priority
    total = distinct_score + weight_score + balance_score + danger_score + hash_score
    return float(total)
