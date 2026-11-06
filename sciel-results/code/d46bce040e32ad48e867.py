"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def priority(vector, n):
    """
    Computes a priority score for a vector in {0, 1, 2}^n.
    Higher scores indicate vectors that should be added earlier in the greedy process.
    Strategy: Prioritize vectors with balanced counts of 0s, 1s, and 2s.
    This avoids early saturation of specific symbol patterns, allowing for larger cap sets.
    """
    c0 = 0
    c1 = 0
    c2 = 0
    
    # Fast counting
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
            
    # Calculate deviation from ideal balance (n/3 each)
    # Using integer arithmetic to avoid float precision issues and speed up
    # deviation = 3*count - n
    d0 = 3 * c0 - n
    d1 = 3 * c1 - n
    d2 = 3 * c2 - n
    
    # Sum of squared deviations (penalty)
    penalty = d0 * d0 + d1 * d1 + d2 * d2
    
    # Tie-breaker: Deterministic hash to ensure unique ordering for equal penalties
    # This helps break symmetries that might lead to suboptimal greedy choices
    h = 0
    for v in vector:
        h = (h * 131 + v) & 0x7FFFFFFF
        
    # Scale tie-breaker to a small range so it doesn't override the main balance metric
    # Max penalty for n=10 is roughly (10-3.33)^2 * 3 ~ 133.
    # Tie breaker should be < 1.
    tie_breaker = (h % 10000) / 10000.0 * 0.01
    
    # Higher priority for lower penalty
    return -penalty + tie_breaker
# EVOLVE-BLOCK-END
