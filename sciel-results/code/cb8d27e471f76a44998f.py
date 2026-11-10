"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Find the longest interval [start, start+length-1] within [1, 1008]
    such that for every x, y in the interval (including x=y), x*y % 1009 != 1.
    
    Approach:
    1. Precompute modular inverses for 1..1008 modulo 1009.
    2. For each left boundary L, determine the maximum right boundary R such that
       the interval [L, R] contains no modular inverse pairs.
    3. An interval [L, R] is invalid if there exists x in [L, R] such that inv(x) is also in [L, R].
    4. For a fixed L, we calculate the minimum upper bound on R imposed by all x >= L.
       - If inv(x) == x, R must be < x.
       - If inv(x) > x, R must be < inv(x).
       - If inv(x) < x:
         - If inv(x) >= L, then both x and inv(x) are >= L. If R >= x, both are in [L, R]. So R must be < x.
         - If inv(x) < L, then inv(x) is outside [L, R] regardless of R. No constraint.
    5. The maximum valid R for L is min(bounds) - 1.
    6. We maximize R - L + 1.
    """
    p = 1009
    n = p - 1  # 1008
    
    # Compute modular inverses
    inv = [0] * (n + 1)
    for x in range(1, n + 1):
        inv[x] = pow(x, p - 2, p)
    
    best_start = 1
    best_len = 0
    
    # For each L, find the max valid R
    # We use O(N^2) which is fast enough for N=1008
    for L in range(1, n + 1):
        min_bound = n + 1  # Infinity equivalent
        
        for x in range(L, n + 1):
            ix = inv[x]
            if ix == x:
                bound = x
            elif ix > x:
                bound = ix
            else: # ix < x
                if ix >= L:
                    bound = x
                else:
                    bound = n + 1
            
            if bound < min_bound:
                min_bound = bound
        
        max_R = min_bound - 1
        if max_R >= L:
            length = max_R - L + 1
            if length > best_len:
                best_len = length
                best_start = L
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
