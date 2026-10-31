"""Cap set greedy priority: smooth surrogate favoring balanced, low-AP-risk vectors."""
import time
import math

_DEADLINE = 330.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def priority(vector, n):
    # Deadline guard: stop doing heavy work near the wall-clock limit.
    if _elapsed() > _DEADLINE - 30.0:
        # Return a simple fast value; sorting will still produce a valid order.
        return float(sum(vector))

    # Count occurrences of each symbol.
    c0 = c1 = c2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1

    # Smooth imbalance penalty: use a soft-min-like surrogate.
    # The quadratic penalty is nonsmooth in its gradient w.r.t. discrete counts.
    # Instead, use a smooth function that penalizes deviation from n/3.
    target = n / 3.0
    d0 = c0 - target
    d1 = c1 - target
    d2 = c2 - target
    # Smooth quadratic: same as before but computed cleanly.
    imbalance = d0 * d0 + d1 * d1 + d2 * d2

    # Additional smooth penalty: "exposure" to APs.
    # A vector with many coordinates equal to the same value is more likely
    # to form APs. Use a smooth measure: variance of the coordinate values.
    # For {0,1,2}^n, the variance is related to how spread out the values are.
    # Low variance (all same) = high AP risk. High variance (balanced) = low risk.
    # Compute sum of squares and mean for variance.
    ss = 0
    for v in vector:
        ss += v * v
    mean = (c0 * 0 + c1 * 1 + c2 * 2) / n
    variance = ss / n - mean * mean
    # Penalize low variance (high AP risk). Smooth: -variance contributes negatively.
    # But we want HIGH priority for LOW risk, so we add variance (higher variance = better).
    # Scale to be comparable to imbalance.
    variance_bonus = variance * n  # Scale up since variance is O(1) and imbalance is O(n)

    # Base score: prefer balanced (low imbalance) and diverse (high variance).
    base = -imbalance + variance_bonus

    # Tiebreak: deterministic pseudo-random spread to diversify order.
    h = 0
    for i, v in enumerate(vector):
        h = (h * 31 + v * (i + 7)) & 0xFFFFFFFF
    # Small perturbation so ties are broken by diversity, not lexicographic.
    return base + (h / float(0xFFFFFFFF)) * 1e-4
