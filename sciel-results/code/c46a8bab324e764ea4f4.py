import time

_START_TIME = time.time()
_TIME_LIMIT = 280.0  # Safety margin

def priority(vector, n):
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        return 0.0
    
    # Fast symbol count
    n0 = 0
    n1 = 0
    n2 = 0
    for v in vector:
        if v == 0:
            n0 += 1
        elif v == 1:
            n1 += 1
        else:
            n2 += 1
    
    # Primary: minimize count of 1s.
    # In {0,1,2}^n, a vector with more 1s is more likely to serve as a midpoint
    # in a 3-term AP (a+c ≡ 2b mod 3). Vectors far from the "center" (all-1s)
    # are safer to add early in greedy construction.
    midpoint_penalty = n1 * n1  # Quadratic to strongly penalize many 1s
    
    # Secondary: reward balance between 0s and 2s for structural diversity.
    # Vectors with well-mixed 0s and 2s spread out in the cube, reducing
    # local clustering and AP formation.
    balance_02 = abs(n0 - n2)
    diversity_bonus = -balance_02
    
    # Tertiary: squared distance from center (all-1s vector)
    # This reinforces the midpoint penalty with a smoother signal.
    dist_sq = n0 + n2  # = n - n1, but using n0+n2 makes it explicit
    
    # Combine: midpoint_penalty dominates, then diversity, then distance
    # Scale factors chosen so that:
    # - midpoint_penalty (range 0 to n^2) dominates
    # - diversity_bonus (range -n to 0) is secondary
    # - dist_sq (range 0 to n) provides fine-grained ordering
    score = -midpoint_penalty * 10.0 + diversity_bonus * 2.0 + dist_sq
    
    # Deterministic tiebreaker via hash
    h = 0
    for i, v in enumerate(vector):
        h = (h * 31 + v * (i + 1) + 7) & 0xFFFFFFFF
    hash_score = (h % 1000000) / 1000000.0
    
    return score + hash_score
