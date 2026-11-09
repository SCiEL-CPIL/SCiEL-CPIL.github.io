"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def priority(vector, n):
    """
    Priority for greedy cap set construction in {0,1,2}^n.
    
    Key insight: Vectors that are "balanced" (coordinates spread across 0,1,2)
    are more likely to form 3-term APs with existing pairs, so they're dangerous
    to add early. Vectors with "polarized" coordinates (clustered near 0 or 2)
    are safer to add first. We also prefer lower weight and use variance as a
    structural measure to spread out the accepted set.
    
    Strategy:
    1. Primary: low variance of coordinates (polarized vectors first)
    2. Secondary: low Hamming weight (sparse vectors)
    3. Tertiary: deterministic hash-based tie-break for stability
    """
    weight = 0
    sum_coords = 0
    sum_sq = 0
    
    for x in vector:
        if x != 0:
            weight += 1
        sum_coords += x
        sum_sq += x * x
    
    # Variance: measures how spread out coordinates are
    # Low variance = polarized (many 0s or many 2s) = safer to add early
    # High variance = balanced = more dangerous (likely AP middle)
    mean = sum_coords / n
    variance = sum_sq / n - mean * mean
    
    # Deterministic tie-break using a simple polynomial hash
    # This avoids Python's hash() which can vary between runs
    h = 0
    for x in vector:
        h = (h * 31 + x) & 0x7FFFFFFF
    jitter = (h % 100000) / 100000.0
    
    # Higher priority = picked first
    # We want: low variance (polarized), low weight, then deterministic order
    # Scale variance down since it's in [0, 4/9] range for {0,1,2}
    score = -(variance * 50.0 + weight * 10.0 + jitter * 0.1)
    
    return score
# EVOLVE-BLOCK-END
