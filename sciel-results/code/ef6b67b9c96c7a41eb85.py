import math
import time

def construct():
    """
    Finds m in [0, 1000000] maximizing f(12, m).
    f(12, m) is the least positive L such that (m, m+L] contains distinct b_1..b_12 with i|b_i.
    """
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    n = 12
    # LCM of 1..12
    K = 1
    for i in range(1, n + 1):
        K = K * i // math.gcd(K, i)
    
    best_m = 0
    best_f = 0
    
    # We know f(12, m) is periodic with period K
    # So we only need to check m in [0, K-1]
    # But we must return an m in [0, 1000000]
    # Any m congruent to best_m mod K works.
    # We will pick the smallest such m >= 0.
    
    # Precompute for each residue r in [0, K-1], and for each L,
    # whether a matching exists. We do this by iterating m from 0 to K-1.
    
    # For speed, we can use a smarter approach:
    # For each m, we need to find the minimum L.
    # Since L is at most 144, we can try L from 1 to 144.
    # But we can be smarter: for each m, the constraint is that
    # for each i in 1..12, there must be a multiple of i in (m, m+L]
    # and all chosen multiples must be distinct.
    
    # Key insight: for a fixed m, the set of multiples of i in (m, m+L] is
    # {ceil((m+1)/i)*i, ceil((m+1)/i)*i + i, ...} up to m+L.
    # So the number of multiples of i in (m, m+L] is floor((m+L)/i) - floor(m/i).
    
    # For a quick necessary condition: for each i, we need at least 1 multiple.
    # Also, the total number of available "slots" must be >= 12.
    
    # Let's use a bipartite matching approach with bitmasks for speed.
    
    for m in range(K):
        if time.time() > deadline:
            break
            
        # Find least L for this m
        for L in range(1, 145):
            if _can_match_fast(m, L, n):
                if L > best_f:
                    best_f = L
                    best_m = m
                break
        
    return best_m

def _can_match_fast(m, L, n):
    """
    Checks if there is a perfect matching for indices 1..n
    in the interval (m, m+L] using bitmasks.
    """
    adj_masks = [0] * n
    
    # Precompute masks for each index
    # bit j-1 corresponds to number m+j
    for j in range(1, L + 1):
        num = m + j
        bit = 1 << (j - 1)
        for i in range(1, n + 1):
            if num % i == 0:
                adj_masks[i - 1] |= bit
                
    # Order indices by number of choices (popcount)
    indices = list(range(n))
    indices.sort(key=lambda x: bin(adj_masks[x]).count('1'))
    
    def dfs(idx, used_mask):
        if idx == n:
            return True
        i = indices[idx]
        avail = adj_masks[i] & ~used_mask
        while avail:
            bit = avail & -avail
            avail ^= bit
            if dfs(idx + 1, used_mask | bit):
                return True
        return False
        
    return dfs(0, 0)
