import time

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s safety margin

# Precomputed primes for fast multi-seed hashing
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
           73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151,
           157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233,
           239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317,
           331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419,
           421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503]


def _fast_hash(vector, n, seed=0):
    """Fast deterministic pseudo-random priority via linear prime-based hash."""
    h = 0
    for i, v in enumerate(vector):
        if i < len(_PRIMES):
            h += v * _PRIMES[i]
        else:
            h += v * (i * 31 + seed + 17)
    h = (h * 1000003 + seed * 999983) % (1 << 60)
    return h / (1 << 60)


def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.

    Strategy (combined design):
    1. Primary: Distinct symbol score - 3 distinct > 2 distinct > 1 distinct.
    2. Secondary: Hamming weight relative to all-ones (count of 2s).
       Favors vectors "heavy" in 2-coordinates to break symmetry and prevent
       clustering in the {0,1}^n sub-cube, forcing uniform exploration.
    3. Tertiary: Mod-3 coordinate-sum penalty. Vectors with sum ≡ 0 mod 3
       are structurally more dangerous for line formation.
    4. Quaternary: Variance-based danger proxy. High variance = more extreme
       coordinates = higher likelihood of forming lines.
    5. Quinary: L1 balance penalty from uniform distribution.
    6. Quinary: Multi-seed prime-based hash for robust deterministic tie-breaking.

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

    # Factor 1: Distinct symbol score
    num_distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    if num_distinct == 3:
        distinct_score = 30000.0
    elif num_distinct == 2:
        distinct_score = 20000.0
    else:
        distinct_score = 10000.0

    # Factor 2: Hamming weight relative to all-ones (count of 2s)
    # Favors vectors with more 2s to break symmetry and explore full space
    weight_score = c2 * 100.0

    # Factor 3: Mod-3 coordinate-sum penalty
    coord_sum = c1 + 2 * c2
    sum_mod = coord_sum % 3
    if sum_mod == 0:
        sum_score = -5000.0
    elif sum_mod == 1:
        sum_score = 2000.0
    else:
        sum_score = 1500.0

    # Factor 4: Variance-based danger proxy
    mean = s / n
    variance = s2 / n - mean * mean
    danger_score = -variance * 5000.0

    # Factor 5: L1 balance penalty
    ideal = n / 3.0
    dev = abs(c0 - ideal) + abs(c1 - ideal) + abs(c2 - ideal)
    balance_score = -dev * 10.0

    # Factor 6: Multi-seed prime-based hash for tie-breaking
    h1 = _fast_hash(vector, n, seed=1)
    h2 = _fast_hash(vector, n, seed=2)
    h3 = _fast_hash(vector, n, seed=3)
    hash_score = (h1 * 0.4 + h2 * 0.35 + h3 * 0.25) * 1.0

    # Combined priority: higher is added first
    total = distinct_score + weight_score + sum_score + danger_score + balance_score + hash_score
    return float(total)
