import time

# EVOLVE-BLOCK-START
_start_time = None
_DEADLINE = 300 * 0.95  # 285s safety margin before the 300s kill

def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.
    
    Strategy: Hybrid Structural-Hash Priority
    1. Macro-Structural Score: 
       - Quadratic distinctness: Strongly prefer vectors using all 3 symbols.
       - Balance Penalty: Penalize deviation from uniform symbol distribution.
       - Variance Penalty: Penalize high coordinate variance (extreme values).
       This creates a smooth gradient favoring "geometrically safe" vectors.
       
    2. Micro-Structural Hash:
       - A polynomial rolling hash of the vector sequence.
       - Unlike a simple tie-breaker, this hash is modulated by the "safety" of the vector.
       - For highly balanced vectors (low risk), the hash weight is higher, encouraging
         diverse exploration within the safe manifold.
       - For unbalanced vectors (high risk), the hash weight is lower, relying more on
         the macro-score to order them, which helps avoid getting stuck in bad local
         configurations of risky vectors.
       
    This combination allows the greedy algorithm to:
    - First, select the most structurally sound vectors.
    - Second, within those, explore a wide variety of patterns via the hash.
    - Third, handle the remaining risky vectors in a deterministic, structurally-aware order.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        # Past deadline: ordering no longer matters, return a flat constant.
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

    # 1. Macro-Structural Score
    # Factor 1: Quadratic distinctness
    num_distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    distinct_score = 10000.0 * num_distinct * num_distinct

    # Factor 2: Balance Penalty (L2 deviation from n/3)
    ideal = n / 3.0
    dev_sq = (c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2
    balance_score = -dev_sq * 10.0  # Increased weight to emphasize balance

    # Factor 3: Variance Penalty
    mean = s / n
    variance = s2 / n - mean * mean
    danger_score = -variance * 500.0

    macro_score = distinct_score + balance_score + danger_score

    # 2. Micro-Structural Hash
    # Polynomial rolling hash
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    
    # Normalize hash to [0, 1]
    hash_norm = h / 0x7FFFFFFF

    # Modulate hash weight based on "safety" (inverse of risk)
    # Risk is approximated by balance_score (negative) and danger_score (negative).
    # We want the hash to be more influential when the vector is "safe" (high macro_score relative to max).
    # A simple proxy: if balance_score is close to 0 (good), weight hash more.
    # balance_score is <= 0. Let's map it to a weight between 0.1 and 1.0.
    # Max possible negative balance_score is roughly n^2/3.
    # For small n, this is manageable.
    # Let's use a sigmoid-like function or simple linear scaling.
    # If dev_sq is small, weight is high.
    # weight = 1.0 / (1.0 + dev_sq * 0.1)
    # This ensures weight is in (0, 1].
    hash_weight = 1.0 / (1.0 + dev_sq * 0.05)
    
    # Scale hash contribution to be significant but not dominant
    micro_score = hash_norm * 1000.0 * hash_weight

    # Combine
    total = macro_score + micro_score
    
    return float(total)
# EVOLVE-BLOCK-END
