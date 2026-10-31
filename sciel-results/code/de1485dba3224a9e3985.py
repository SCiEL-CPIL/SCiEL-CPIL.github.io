import time

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s safety margin


def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.

    Strategy (combined design, smooth surrogates):
    1. Primary: Smooth quadratic distinct-symbol score. Quadratic in the number
       of present symbols (0,1,4,9 * 10000) gives a smoother landscape than the
       discrete 10000/20000/30000 step, producing a better gradient signal for
       the greedy ordering.
    2. Secondary: Hamming weight relative to all-ones (count of 2s). Favors
       vectors "heavy" in the 2-coordinate to break the symmetry of the ternary
       grid and prevent the greedy algorithm from clustering in the {0,1}^n
       sub-cube, forcing uniform exploration of the full {0,1,2}^n space.
    3. Tertiary: Quadratic balance penalty. L2 deviation from the uniform
       (n/3, n/3, n/3) distribution, a smooth surrogate for the L1 deviation,
       provides a smoother gradient than the piecewise-linear L1 penalty.
    4. Quaternary: Smooth variance-based danger proxy. High coordinate variance
       means more "extreme" coordinates, which are structurally more likely to
       complete an affine line, so we penalize variance.
    5. Quinary: Fast polynomial rolling hash for robust deterministic
       tie-breaking.

    All components computed in O(n) time with minimal per-call overhead.
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
    # Smoother than the discrete step, gives a better gradient for the ordering.
    num_distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    distinct_score = 10000.0 * num_distinct * num_distinct

    # Factor 2: Hamming weight relative to all-ones (count of 2s).
    # Favors vectors with more 2s to break symmetry and explore the full space,
    # preventing clustering in the {0,1}^n sub-cube.
    weight_score = c2 * 100.0

    # Factor 3: Quadratic balance penalty (smooth surrogate for L1).
    # L2 deviation from the uniform (n/3, n/3, n/3) distribution.
    ideal = n / 3.0
    dev_sq = (c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2
    balance_score = -dev_sq * 5.0

    # Factor 4: Smooth variance-based danger proxy.
    # High coordinate variance = more extreme coordinates = higher line risk.
    mean = s / n
    variance = s2 / n - mean * mean
    danger_score = -variance * 5000.0

    # Factor 5: Fast polynomial rolling hash for deterministic tie-breaking.
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = h / 0x7FFFFFFF

    # Combined priority: higher is added first.
    total = distinct_score + weight_score + balance_score + danger_score + hash_score
    return float(total)
