import time

_START_TIME = time.time()
_TIME_LIMIT = 280.0  # Safety margin

def priority(vector, n):
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        return 0.0

    if n <= 0:
        return 0.0

    # 1. Compute Composition Counts and Structural Features in one pass
    c0 = 0
    c1 = 0
    c2 = 0
    sq_edge = 0
    s = 0
    s2 = 0
    
    for x in vector:
        if x == 0:
            c0 += 1
            # sq_edge += (0-1)^2 = 1
            sq_edge += 1
            # s += 0, s2 += 0
        elif x == 1:
            c1 += 1
            # sq_edge += (1-1)^2 = 0
            s += 1
            s2 += 1
        else:
            c2 += 1
            # sq_edge += (2-1)^2 = 1
            sq_edge += 1
            s += 2
            s2 += 4

    # 2. Balance Score (Primary)
    # From Inspiration 1: Minimize L1 distance from perfect balance (n/3, n/3, n/3)
    # This targets the high-density region of the cap set.
    # d = |c0-c1| + |c1-c2| + |c0-c2|
    # Normalized to [0, 1] where 1 is perfectly balanced.
    # Max possible d is 2n (e.g., all 0s vs all 2s? No, max diff is n. 
    # If c0=n, c1=0, c2=0 -> d = n + 0 + n = 2n. Correct.)
    d = abs(c0 - c1) + abs(c1 - c2) + abs(c0 - c2)
    balance_score = 1.0 - (d / (2.0 * n)) if n > 0 else 0.0
    
    # 3. Squared Edge Score (Secondary)
    # From Inspiration 2: Reward vectors far from the center (value 1).
    # Vectors with many 0s and 2s are less likely to be midpoints of 3-APs.
    # Normalized to [0, 1].
    sq_edge_norm = sq_edge / n if n > 0 else 0.0

    # 4. Variance Score (Tertiary)
    # From Inspiration 2: Structural diversity.
    mean = s / n
    var = (s2 / n) - mean * mean
    if var < 0:
        var = 0.0
    # Max variance for {0,1,2} is 2/3 (if half are 0 and half 2, mean 1, var 1? 
    # E[x^2] - (E[x])^2. If 50% 0, 50% 2: E[x]=1, E[x^2]=2, var=1. 
    # If 1/3 each: E[x]=1, E[x^2]=2/3, var=2/3.
    # Max var is 1.0.
    var_norm = var / 1.0

    # 5. Diversity Bonus (Quaternary)
    # From Inspiration 2: Reward using all three symbols.
    diversity = 0.0
    if c0 > 0 and c1 > 0 and c2 > 0:
        diversity = 1.0
    elif (c0 > 0 and c1 > 0) or (c0 > 0 and c2 > 0) or (c1 > 0 and c2 > 0):
        diversity = 0.5
    else:
        diversity = 0.25

    # 6. Non-linear Hash Tie-breaker (Quinary)
    # From Cell Findings: Non-linear bit mixing (MurmurHash3 finalizer) breaks spatial correlation.
    # This is crucial to scatter "AP-prone" neighbors that might have similar structural scores.
    
    # Convert vector to a base-3 integer representation for hashing? 
    # Or just mix the values directly with a strong finalizer.
    # Let's use a simple but effective mixing function on the raw bytes/values.
    
    h = 0x811c9dc5  # FNV offset basis
    for i, x in enumerate(vector):
        h ^= x
        h = (h * 0x01000193) & 0xFFFFFFFF
        
    # MurmurHash3 finalizer (mix)
    h ^= (h >> 16)
    h = (h * 0x85ebca6b) & 0xFFFFFFFF
    h ^= (h >> 13)
    h = (h * 0xc2b2ae35) & 0xFFFFFFFF
    h ^= (h >> 16)
    
    # Normalize to [0, 1)
    hash_score = (h & 0xFFFFFFFF) / 4294967296.0

    # Combine Scores
    # We want balance to dominate, but the non-linear hash needs to be strong enough
    # to break ties among vectors with similar structural properties.
    # Since structural scores are in [0,1], we can weight them directly.
    
    # Weights:
    # Balance: 1.0 (Dominant)
    # Sq Edge: 0.2 (Secondary structural preference)
    # Variance: 0.1 (Tertiary)
    # Diversity: 0.1 (Quaternary)
    # Hash: 0.01 (Tie-breaker, but scaled up slightly to ensure it's not lost in float precision 
    #           if structural scores are very close. However, since structural scores are coarse 
    #           (based on counts), many vectors will have identical structural scores. 
    #           The hash MUST be the deciding factor for those.)
    
    # Actually, if structural scores are identical, we rely entirely on the hash.
    # If we add 0.01 * hash, and the difference in structural score is > 0.01, hash is ignored.
    # If structural score is identical, hash decides.
    # This is the desired behavior.
    
    score = (
        1.0 * balance_score
        + 0.2 * sq_edge_norm
        + 0.1 * var_norm
        + 0.1 * diversity
        + 0.01 * hash_score
    )

    return float(score)
