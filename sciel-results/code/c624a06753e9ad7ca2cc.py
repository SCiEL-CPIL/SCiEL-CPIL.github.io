import time
import math

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s safety margin
_log = math.log


def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.

    Strategy (refined within distinct-count + entropy + hash family):
    1. Primary: Quadratic distinct-symbol score (dominant, O(1)).
    2. Secondary: Shannon entropy of symbol distribution (smooth balance).
    3. Tertiary: Mild preference for vectors with at least one 2 (breaks
       {0,1}^n sub-cube clustering without over-weighting).
    4. Quaternary: Polynomial rolling hash for deterministic tie-breaking.

    All components computed in O(n) time with minimal per-call overhead.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        return 0.0

    # Fast symbol counting in one pass
    c0 = 0
    c1 = 0
    c2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:  # v == 2
            c2 += 1

    # Factor 1: Quadratic distinct-symbol score.
    num_distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    distinct_score = 10000.0 * num_distinct * num_distinct

    # Factor 2: Shannon entropy of symbol distribution.
    # Higher entropy = more uniform spread = lower line-completion risk.
    entropy = 0.0
    for c in (c0, c1, c2):
        if c > 0:
            p = c / n
            entropy -= p * _log(p)
    entropy_score = entropy * 500.0

    # Factor 3: Mild preference for having at least one 2 to break symmetry.
    # Using a soft indicator: c2 / (c2 + 1) saturates at 1 for c2 >= 1.
    symmetry_break = (c2 / (c2 + 1.0)) * 10.0

    # Factor 4: Fast polynomial rolling hash for deterministic tie-breaking.
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = h / 0x7FFFFFFF

    total = distinct_score + entropy_score + symmetry_break + hash_score
    return float(total)
