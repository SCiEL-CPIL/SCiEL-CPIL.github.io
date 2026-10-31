import time

_start_time = None
_DEADLINE = 330 * 0.95

def priority(vector, n):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        return 0.0

    c0 = 0
    c1 = 0
    c2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1

    # 1. Distinct symbol score
    num_distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    distinct_score = 10000.0 * num_distinct * num_distinct

    # 2. Balance score (L2 deviation from uniform)
    ideal = n / 3.0
    dev_sq = (c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2
    balance_score = -dev_sq * 5.0

    # 3. Spectral Energy Penalty
    # Energy E = sum x_i^2 (in R) = 1*c1 + 4*c2
    # Penalize high energy to favor "low frequency" (sparse) vectors
    energy = c1 + 4.0 * c2
    spectral_penalty = -energy * 10.0

    # 4. Phase Imbalance Penalty
    # Imbalance between 1s and 2s
    phase_diff = c1 - c2
    phase_penalty = -phase_diff * phase_diff * 2.0

    # 5. Hash
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = h / 0x7FFFFFFF

    total = distinct_score + balance_score + spectral_penalty + phase_penalty + hash_score
    return float(total)
