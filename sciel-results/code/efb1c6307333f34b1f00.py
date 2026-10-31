# EVOLVE-BLOCK-START
import time

# Cache for priority computation to avoid redundant work
_priority_cache = {}

def priority(vector, n):
    """
    Prioritize vectors based on geometric conflict potential.
    
    A vector v is "safe" if it forms few affine lines with other vectors.
    In F_3^n, three points a, b, c form an affine line if a + b + c = 0 (mod 3).
    For a given vector v, the number of potential lines it can form depends on
    how many pairs (u, w) satisfy u + w = -v (mod 3) component-wise.
    
    We approximate this by counting how many vectors u in {0,1,2}^n have the
    property that -v - u is also in {0,1,2}^n (always true), but weight by
    how "central" v is - vectors near the center of the cube have more potential
    partners.
    
    Additionally, we penalize vectors that are themselves midpoints of many
    potential lines, as these create more constraints.
    """
    # Use a hashable key for caching
    key = (tuple(vector), n)
    if key in _priority_cache:
        return _priority_cache[key]
    
    # Compute the number of "conflict partners" - vectors u such that
    # there exists w with u + w + v = 0 mod 3.
    # For each coordinate, if v[i] = 0, then u[i] + w[i] = 0, so u[i] = w[i]
    # if v[i] = 1, then u[i] + w[i] = 2, so (u[i], w[i]) in {(0,2),(1,1),(2,0)}
    # if v[i] = 2, then u[i] + w[i] = 1, so (u[i], w[i]) in {(0,1),(1,0),(2,2)}
    # The number of valid (u,w) pairs per coordinate is always 3, so total is 3^n.
    # This doesn't differentiate vectors. Instead, we use a different heuristic.
    
    # Heuristic: Count how many "midpoint positions" v occupies.
    # A vector v is a midpoint of a line if there exists d such that
    # v-d and v+d are in {0,1,2}^n. The number of such d depends on v.
    # For coordinate i:
    #   if v[i] = 0: d[i] can be 0 (gives 0,0) or 1 (gives 2,1) or 2 (gives 1,2) -> 3 choices
    #   if v[i] = 1: d[i] can be 0 (gives 1,1) or 1 (gives 0,2) or 2 (gives 2,0) -> 3 choices  
    #   if v[i] = 2: d[i] can be 0 (gives 2,2) or 1 (gives 1,0) or 2 (gives 0,1) -> 3 choices
    # Again 3^n for all. Need a better approach.
    
    # Better heuristic: Use the "energy" of the vector - how close it is to
    # being a constant vector. Constant vectors (all 0s, all 1s, all 2s) are
    # more likely to form lines with many other vectors.
    # Also, vectors with more 1s are more likely to be midpoints.
    
    n1 = 0
    n2 = 0
    for x in vector:
        if x == 1:
            n1 += 1
        elif x == 2:
            n2 += 1
    n0 = n - n1 - n2
    
    # Penalty for being "central" - vectors with balanced counts are more central
    # and thus have more potential line partners
    target = n / 3.0
    centrality = (n0 - target)**2 + (n1 - target)**2 + (n2 - target)**2
    # Higher centrality means LESS balanced, so we want to MINIMIZE this
    # But wait - we want to AVOID vectors that form many lines, so we should
    # PENALIZE well-balanced vectors (low centrality deviation)
    
    # Additional penalty: vectors with many 1s are more likely to be midpoints
    # because 1 is the "middle" value in F_3
    midpoint_risk = n1 * 2.0  # Weight 1s more heavily
    
    # Also penalize vectors that are close to other common vectors
    # Use a simple proxy: variance of the coordinate values
    # Low variance = more uniform = potentially more line partners
    variance = (n0 * 0 + n1 * 1 + n2 * 2) / n if n > 0 else 0
    variance = (n0 * 0 + n1 * 1 + n2 * 4) / n - variance**2 if n > 0 else 0
    
    # Combined score: we want to MINIMIZE line formation potential
    # So we return the NEGATIVE of the risk score
    risk = centrality * 0.5 + midpoint_risk * 1.0 - variance * 0.3
    result = -risk
    
    _priority_cache[key] = result
    return result
# EVOLVE-BLOCK-END
