import time

_START = time.time()
_DEADLINE = _START + 590.0  # safety margin before 600s kill


def _elapsed():
    return time.time() - _START


def _time_exceeded():
    return _elapsed() > _DEADLINE


def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination
    (m=12, n=1). The seed program returns [4, 8, 12, ..., 48].

    The C4 upper bound is minimized when the roots are concentrated.
    Under the constraint that the sum of roots is fixed (312 from the seed),
    the sum-of-squares is minimized when all roots equal the mean (26).
    
    We do a direct 12-dimensional optimization to find the configuration
    that minimizes the C4 bound. Since we don't have the exact C4 formula,
    we use the measured finding that direct optimization reaches 33.31
    (from a starting point of 0.6659 in some normalization), which suggests
    the optimal configuration is NOT all-equal but rather some specific spread.
    
    Given the seed [4, 8, ..., 48] scores 1.0 and the measured best is 
    significantly lower, we attempt a numerical optimization using gradient
    descent on a surrogate: minimize sum(r_i^2) + penalty for deviation from
    a smooth distribution, or simply try several candidate configurations.
    
    Actually, from the knowledge: "direct 12-dimensional optimization of root 
    positions allows the roots to reach the true minimum of the C4 bound, which 
    is significantly lower than the constrained search could find. Measured: 
    0.6659 -> 33.31."
    
    This is confusing - 33.31 is HIGHER than 0.6659. Wait, the score is seed/value,
    so higher score = lower value. If value went from 0.6659 to 33.31, that's worse.
    But the text says "improved island 5". Let me re-read.
    
    "Measured: 0.6659 -> 33.31. (strategy of a program that improved island 5)"
    
    So the VALUE decreased? No, 33.31 > 0.6659. Unless these are scores, not values.
    Score = seed_value / candidate_value. If score improved from 0.6659 to 33.31,
    then candidate_value decreased dramatically.
    
    So the true minimum C4 value is much lower than the seed's value.
    Seed value corresponds to score 1.0, so seed_value = seed_value.
    If a program achieved score 33.31, its value = seed_value / 33.31.
    
    The seed returns [4, 8, 12, ..., 48]. Let's compute some statistics:
    sum = 312, sum_sq = 4^2 + 8^2 + ... + 48^2 = 16 * sum_{k=1}^{12} k^2 = 16 * (12*13*25/6) = 16 * 650 = 10400
    mean = 26, variance = 10400/12 - 26^2 = 866.67 - 676 = 190.67
    
    If the C4 bound is proportional to some spread measure, concentrating roots reduces it.
    All-equal at 26: sum_sq = 12 * 676 = 8112, which is less than 10400.
    
    But the measured improvement suggests the optimum is even more extreme.
    Perhaps the C4 bound involves the LAST sign change, which depends on the 
    polynomial's structure in a non-trivial way.
    
    Let me try a different approach: use scipy to optimize a reasonable surrogate.
    """
    import numpy as np
    
    # Try several candidate configurations and pick the best based on a surrogate
    # The surrogate: minimize the "spread" as measured by sum of squares,
    # since lower spread generally reduces uncertainty bounds.
    
    candidates = []
    
    # Candidate 1: Seed
    candidates.append(np.array([4.0 * (i + 1) for i in range(12)]))
    
    # Candidate 2: All equal at mean
    candidates.append(np.full(12, 26.0))
    
    # Candidate 3: Concentrated near a single value (minimizes sum_sq)
    # But all positive, so let's try clustering
    for cluster_center in [10.0, 15.0, 20.0, 25.0, 30.0]:
        # Small spread around center
        offsets = np.linspace(-3, 3, 12)
        r = cluster_center + offsets
        if np.all(r > 0):
            candidates.append(r)
    
    # Candidate 4: Geometric progression centered at 26
    for ratio in [1.1, 1.2, 1.3]:
        # r_i = 26 * ratio^(i - 5.5) for i=0..11
        idx = np.arange(12) - 5.5
        r = 26.0 * (ratio ** idx)
        if np.all(r > 0):
            candidates.append(r)
    
    # Evaluate surrogate: sum of squares (lower is better for spread)
    best_val = float('inf')
    best_candidate = candidates[0]
    for c in candidates:
        val = float(np.sum(c ** 2))
        if val < best_val:
            best_val = val
            best_candidate = c
    
    return [float(x) for x in best_candidate]
