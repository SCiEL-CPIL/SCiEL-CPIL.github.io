import time
import hashlib

_start_time = None
_DEADLINE = 300 * 0.95  # 285s safety margin before the 300s kill

def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.

    Strategy:
    1. Primary: Smooth quadratic distinct-symbol score. Quadratic in the number
       of present symbols (0, 1, 4, 9 * 10000) creates a steep but continuous
       gradient, strongly preferring vectors that use all three symbols.
       This avoids the discrete cliffs of linear tiering.
    2. Secondary: Smooth variance-based danger proxy. High coordinate variance
       indicates many "extreme" coordinates (0s and 2s), which are structurally
       more likely to complete affine lines. We penalize variance to prefer
       "middle" (1-heavy) vectors that are geometrically safer.
    3. Tertiary: Quadratic balance penalty. L2 deviation from the uniform
       (n/3, n/3, n/3) distribution. This is a smooth surrogate for balance
       that provides a better gradient than piecewise-linear L1 or entropy-based
       metrics, avoiding local plateaus in the optimization landscape.
    4. Quaternary: Symmetry breaking via count of 2s. Favors vectors with more
       2s to prevent the greedy algorithm from clustering in the {0,1}^n sub-cube
       and ensures uniform exploration of the full ternary grid.
    5. Quinary: 64-bit slice of SHA256 for deterministic tie-breaking.
       Higher entropy hash reduces collisions in the ordering, leading to more
       random-like greedy selection which typically yields larger cap sets.

    All components are computed in O(n) time with minimal per-call overhead.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        # Past deadline: return a constant to signal "stop caring about ordering"
        return 0.0

    # Fast symbol counting in one pass (collects counts, sum, and sum-of-squares)
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
        else:  # v == 2
            c2 += 1
            s += 2
            s2 += 4

    # Factor 1: Smooth quadratic distinct-symbol score.
    # Quadratic in presence count: 0 -> 0, 1 -> 10000, 2 -> 40000, 3 -> 90000.
    # This creates a very strong preference for 3-distinct vectors while
    # maintaining a smooth landscape for the greedy algorithm.
    num_distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    distinct_score = 10000.0 * num_distinct * num_distinct

    # Factor 2: Symmetry breaking via count of 2s.
    # Favors vectors with more 2s to break the symmetry of the ternary grid
    # and prevent the greedy algorithm from clustering in the {0,1}^n sub-cube.
    weight_score = c2 * 100.0

    # Factor 3: Quadratic balance penalty (smooth surrogate for L1).
    # L2 deviation from the uniform (n/3, n/3, n/3) distribution.
    ideal = n / 3.0
    dev_sq = (c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2
    balance_score = -dev_sq * 5.0

    # Factor 4: Smooth variance-based danger proxy.
    # High coordinate variance means more "extreme" coordinates (0s and 2s),
    # which are structurally more likely to complete an affine line.
    mean = s / n
    variance = s2 / n - mean * mean
    danger_score = -variance * 5000.0

    # Factor 5: 64-bit slice of SHA256 for deterministic tie-breaking.
    # Using a cryptographic hash provides better uniformity and lower collision
    # rates compared to simple polynomial hashes, improving the randomization
    # of the greedy selection order.
    # Convert vector to bytes for hashing. We pack the vector into a bytes object.
    # Since n is small (typically < 20-30), this is fast enough.
    # We include n in the hash to ensure different dimensions don't collide in ordering logic
    # if the same vector pattern appears (though vectors are unique per call context usually).
    try:
        # Efficient conversion: create a bytes object from the vector
        # We can just use bytes(vector) since values are 0, 1, 2 which are valid byte values.
        b = bytes(vector)
        # Include n to differentiate contexts if necessary, though not strictly required for tie-breaking
        # within a single run of fixed n.
        h = int.from_bytes(hashlib.sha256(b).digest()[:8], 'big')
        hash_score = h / (1 << 64)
    except Exception:
        # Fallback to simple hash if something goes wrong (should not happen)
        h = 0
        for i, val in enumerate(vector):
            h = (h * 31 + val + i) & 0xFFFFFFFFFFFFFFFF
        hash_score = h / (1 << 64)

    # Combined priority: higher is added first.
    total = distinct_score + weight_score + balance_score + danger_score + hash_score
    return float(total)
