"""Cap set greedy builder with smooth quadratic surrogates for line-avoidance."""
import time

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s safety margin


def priority(vector, n):
    """
    Priority for greedy cap set construction in {0,1,2}^n.

    Structural design (all smooth/quadratic for continuous landscape):
    1. Quadratic distinctness: reward having all three symbols present,
       with a smooth penalty for missing symbols via (count/n)² terms.
    2. Smooth mod-3 sum penalty: quadratic in (sum mod 3 distance to 0),
       replacing discrete tiers with a continuous proxy.
    3. Variance-based danger proxy: high variance = more extreme = more line risk.
    4. L2 (quadratic) balance penalty: smooth penalty for imbalance in symbol counts.
    5. Fast rolling hash for deterministic tie-breaking.

    All factors are differentiable in the count space, creating a smooth
    landscape that helps the greedy algorithm avoid local plateaus.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        return 0.0

    # Fast symbol counting
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)

    # Factor 1: Quadratic distinctness score
    # Reward for having diverse symbols: sum of (count_i / n)^2 penalizes
    # concentration, so we maximize spread. Use a smooth surrogate:
    # distinct_score = n * (1 - sum((c_i/n)^2)) = n - (c0²+c1²+c2²)/n
    # This is high when all three symbols are well-represented.
    sq_sum = c0 * c0 + c1 * c1 + c2 * c2
    distinct_score = 10000.0 * (n - sq_sum / n)

    # Factor 2: Smooth mod-3 sum penalty
    # coord_sum mod 3 determines line-completion risk.
    # Instead of discrete tiers, use a smooth proxy:
    # distance to nearest multiple of 3, squared, scaled.
    coord_sum = c1 + 2 * c2  # c0*0 + c1*1 + c2*2
    sum_mod = coord_sum % 3
    # Distance from 0 in cyclic group Z_3: 0→0, 1→1, 2→1
    dist_to_zero = 0 if sum_mod == 0 else 1
    # Quadratic penalty: vectors with sum≡0 mod 3 are most dangerous
    # Use a smooth proxy: penalty proportional to (1 - dist)² which is 1 for mod 0, 0 for mod 1/2
    # Better: use cos(2π·sum_mod/3) which is 1 for mod 0, -0.5 for mod 1/2
    # This gives a smooth cyclic penalty
    import math
    sum_score = -5000.0 * math.cos(2.0 * math.pi * sum_mod / 3.0)

    # Factor 3: Variance-based danger proxy
    # High variance = more "extreme" coordinates = more likely to form lines
    # Compute sum and sum-of-squares in one pass
    s = 0
    s2 = 0
    for v in vector:
        s += v
        s2 += v * v
    mean = s / n
    variance = s2 / n - mean * mean
    danger_score = -variance * 5000.0

    # Factor 4: L2 (quadratic) balance penalty
    # Smooth penalty for deviation from ideal count n/3
    ideal = n / 3.0
    dev_sq = (c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2
    balance_score = -dev_sq * 10.0

    # Factor 5: Fast polynomial rolling hash for deterministic tie-breaking
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = h / 0x7FFFFFFF

    # Combined priority: higher is added first
    total = distinct_score + sum_score + danger_score + balance_score + hash_score
    return float(total)
