# EVOLVE-BLOCK-START
import time
import math

_start_time = None
_deadline = 320.0  # safety margin before the 330s kill

def priority(vector, n):
    """
    Priority for greedy cap-set construction in {0,1,2}^n.

    Structural Strategy:
    1. Soft-Max Distinctness: Smooth log-sum-exp surrogate for preferring
       vectors with all 3 symbols. Replaces hard quadratic tiering with a
       continuous gradient that still strongly favors 3-distinct vectors.
    2. L2 Balance Penalty: Penalize deviation from the uniform symbol distribution
       (n/3, n/3, n/3) using squared error.
    3. Variance Penalty: Penalize high coordinate variance (extreme 0/2 coords).
    4. Spectral Smoothness Penalty: Penalize high-frequency coordinate patterns
       (alternating 0/2) that are more likely to form affine lines.
    5. Symmetry Breaking: Favor vectors with more 2s.
    6. Deterministic Hash: Fast polynomial hash for tie-breaking.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _deadline:
        return 0.0

    # Fast symbol counting in one pass (collects counts, sum, sum-of-squares,
    # and spectral smoothness metric)
    c0 = 0
    c1 = 0
    c2 = 0
    s = 0
    s2 = 0
    # Spectral: sum of |v[i+1] - v[i]| as a proxy for high-frequency content
    adj_diff_sum = 0
    prev_val = None
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
        if prev_val is not None:
            adj_diff_sum += abs(v - prev_val)
        prev_val = v

    # Factor 1: Soft-max distinctness via log-sum-exp.
    # For each symbol present, assign a base weight; use log-sum-exp for smoothness.
    # Weights: w0=1, w1=1, w2=1 if present, 0 otherwise.
    # log-sum-exp of [w0*log(A), w1*log(A), w2*log(A)] where A is a large base.
    # Simplified: use a smooth function of the counts that approximates
    # "prefer all 3 symbols" with a continuous gradient.
    # Use: exp(c0/n * alpha) + exp(c1/n * alpha) + exp(c2/n * alpha)
    # where alpha controls sharpness. This smoothly peaks when all counts are high.
    alpha = 8.0
    e0 = math.exp(c0 / n * alpha) if c0 > 0 else 0.0
    e1 = math.exp(c1 / n * alpha) if c1 > 0 else 0.0
    e2 = math.exp(c2 / n * alpha) if c2 > 0 else 0.0
    distinct_score = 10000.0 * (e0 + e1 + e2) / 3.0

    # Factor 2: Symmetry breaking via count of 2s.
    weight_score = c2 * 100.0

    # Factor 3: Quadratic balance penalty.
    ideal = n / 3.0
    dev_sq = (c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2
    balance_score = -dev_sq * 5.0

    # Factor 4: Variance-based danger proxy.
    mean = s / n
    variance = s2 / n - mean * mean
    danger_score = -variance * 6000.0

    # Factor 5: Spectral smoothness penalty.
    # High adjacent-difference sum indicates high-frequency (alternating) patterns
    # that are more prone to forming affine lines with other structured vectors.
    # Normalize by n-1 to get per-edge average, then penalize.
    if n > 1:
        adj_avg = adj_diff_sum / (n - 1)
        spectral_score = -adj_avg * 200.0
    else:
        spectral_score = 0.0

    # Factor 6: Fast polynomial rolling hash for deterministic tie-breaking.
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = h / 0x7FFFFFFF

    # Combined priority: higher is added first.
    total = distinct_score + weight_score + balance_score + danger_score + spectral_score + hash_score
    return float(total)
# EVOLVE-BLOCK-END
