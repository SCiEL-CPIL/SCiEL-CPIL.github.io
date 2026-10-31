"""Cap set greedy builder: simplified hybrid with clean variance penalty.

Key insight from inspirations: The complex spectral/soft-min machinery in the
previous version hurt performance. A clean variance-based danger penalty works
better because it directly measures the "extremeness" of coordinates (0s and 2s)
which correlate with line-completion risk.

Strategy:
1. Quadratic distinctness: Strongly prefer vectors using all 3 symbols.
2. Symmetry breaking: Favor vectors with more 2s.
3. L2 balance penalty: Smooth penalty for deviation from uniform distribution.
4. Variance penalty: Clean penalty for high coordinate variance (extreme coords).
5. Hash tie-breaking: Deterministic fast polynomial hash.
"""
import time

_start_time = None
_n_cache = None
_priority_cache = None
_deadline = 320.0  # Safety margin before the 330s kill


def _init(n):
    global _start_time, _n_cache, _priority_cache
    if _start_time is None:
        _start_time = time.time()
    if _n_cache is None or _n_cache != n:
        _n_cache = n
        _priority_cache = {}


def _priority_core(vec, n):
    """
    Deterministic, vector-intrinsic priority.

    Components:
    1. distinct_score: Quadratic score for using all 3 symbols.
    2. weight_score: Symmetry-breaking bonus for vectors with more 2s.
    3. balance_score: L2 penalty for deviation from uniform symbol distribution.
    4. danger_score: Variance penalty (high variance = more line risk).
    5. hash_score: Fast polynomial hash for deterministic tie-breaking.
    """
    # --- Fast symbol counting in one pass ---
    c0 = 0
    c1 = 0
    c2 = 0
    s = 0
    s2 = 0
    for v in vec:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
            s += 1
            s2 += 1
        else:  # v == 2
            c2 += 1
            s += 2
            s2 += 4

    total = n
    if total == 0:
        return 0.0

    # Factor 1: Smooth quadratic distinct-symbol score.
    # Quadratic in presence count: 0 -> 0, 1 -> 10000, 2 -> 40000, 3 -> 90000.
    num_distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    distinct_score = 10000.0 * num_distinct * num_distinct

    # Factor 2: Symmetry breaking via count of 2s.
    weight_score = c2 * 100.0

    # Factor 3: Quadratic balance penalty (smooth surrogate for L1).
    ideal = n / 3.0
    dev_sq = (c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2
    balance_score = -dev_sq * 5.0

    # Factor 4: Clean variance-based danger proxy.
    # High coordinate variance means more "extreme" coordinates (0s and 2s),
    # which are structurally more likely to complete an affine line.
    mean = s / total
    variance = s2 / total - mean * mean
    danger_score = -variance * 5000.0

    # Factor 5: Fast polynomial rolling hash for deterministic tie-breaking.
    h = 0
    for i, val in enumerate(vec):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = h / 0x7FFFFFFF

    # --- Combine ---
    total_priority = (
        distinct_score
        + weight_score
        + balance_score
        + danger_score
        + hash_score
    )
    
    return float(total_priority)


def priority(vector, n):
    """
    Returns a priority value for the given vector. Higher priority means the
    vector is considered first in the greedy insertion.

    Strategy: Clean hybrid of chaos-breaking (symmetry breaking + distinctness)
    and variance-based danger penalty.
    """
    _init(n)

    # Deadline guard: return a fast deterministic fallback near the limit.
    elapsed = time.time() - _start_time
    if elapsed > _deadline:
        vec = tuple(vector)
        # Fast fallback: just a hash to ensure distinctness and speed
        h = 0
        for i, val in enumerate(vec):
            h = (h * 31 + val + i) & 0x7FFFFFFF
        return float(h)

    vec = tuple(vector)

    if vec in _priority_cache:
        return _priority_cache[vec]

    val = _priority_core(vec, n)
    _priority_cache[vec] = val
    return val
