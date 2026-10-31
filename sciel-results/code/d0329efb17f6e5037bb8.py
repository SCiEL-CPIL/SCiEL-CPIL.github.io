import time

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s safety margin


def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.
    
    Strategy:
    1. Primary: Smooth quadratic distinct-symbol score (reward all 3 symbols).
    2. Secondary: Mod-3 coordinate-sum penalty (structural line-avoidance signal).
    3. Tertiary: Heavy-2 symmetry-breaking bonus (favor vectors with more 2s
       to prevent clustering in {0,1}^n sub-cube).
    4. Quaternary: Quadratic balance penalty (smooth deviation from uniform).
    5. Quinary: Fast rolling hash for deterministic tie-breaking.
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

    # Factor 1: Quadratic distinct symbol score (smooth landscape)
    presence0 = 1.0 if c0 > 0 else 0.0
    presence1 = 1.0 if c1 > 0 else 0.0
    presence2 = 1.0 if c2 > 0 else 0.0
    num_distinct = presence0 + presence1 + presence2
    distinct_score = 10000.0 * num_distinct * num_distinct

    # Factor 2: Mod-3 coordinate-sum penalty
    # In F_3^n, three points form an affine line iff their coordinate-wise sum ≡ 0 mod 3.
    # Vectors with sum ≡ 0 mod 3 are more likely to complete affine lines.
    coord_sum = c1 + 2 * c2  # 0*c0 + 1*c1 + 2*c2
    sum_mod = coord_sum % 3
    if sum_mod == 0:
        sum_score = -4000.0
    elif sum_mod == 1:
        sum_score = 1500.0
    else:
        sum_score = 1000.0

    # Factor 3: Heavy-2 symmetry-breaking bonus
    # Favor vectors with more 2s to break the {0,1}^n clustering bias.
    # This encourages exploration of the full {0,1,2}^n space.
    # Scale relative to n so the bonus is meaningful across dimensions.
    heavy2_bonus = c2 * (n / 3.0) * 2.0

    # Factor 4: Quadratic balance penalty (smoother than L1)
    ideal = n / 3.0
    dev_sq = (c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2
    balance_score = -dev_sq * 5.0

    # Factor 5: Fast polynomial rolling hash for deterministic tie-breaking
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = h / 0x7FFFFFFF

    return float(distinct_score + sum_score + heavy2_bonus + balance_score + hash_score)
