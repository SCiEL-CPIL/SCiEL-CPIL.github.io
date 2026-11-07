import time
from math import gcd

def construct():
    """
    Find m in [0, 1000000] that maximizes f(12, m).
    f(12, m) is the least positive L such that (m, m+L] contains distinct b_1,...,b_12 with i | b_i.
    """
    N = 12
    MAX_M = 1000000
    start_time = time.time()
    DEADLINE = start_time + 75  # 75s safety margin out of 90s
    
    # LCM of 1..12
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    M = 1
    for i in range(1, N + 1):
        M = lcm(M, i)
    # M = 27720
    
    # Precompute divisibility mask for each position offset 1..144
    # For a given r, the number at offset j is (r + j).
    # We need to know for each j in 1..144, which divisors i in 1..12 divide (r+j).
    # This depends on r mod M, but since we're iterating over all r in 0..M-1,
    # we can compute on the fly efficiently.
    
    # Key optimization: For each r, precompute div_mask[j] = bitmask of divisors
    # that divide (r + j), for j = 1..144.
    # Then has_matching(r, L) only needs to check if a perfect matching exists
    # using the first L positions.
    
    def has_matching_from_masks(div_masks, L):
        """
        div_masks: list of length 145 (index 1..144), each is a bitmask of which i in 1..12 divide that position.
        Check if perfect matching exists using positions 1..L.
        """
        # adj[i] = list of positions j in 1..L where i divides position j
        adj = [[] for _ in range(N + 1)]
        for j in range(1, L + 1):
            mask = div_masks[j]
            for i in range(1, N + 1):
                if mask & (1 << (i - 1)):
                    adj[i].append(j)
        
        # Kuhn's algorithm
        match_right = [-1] * (L + 1)
        
        def dfs(i, visited):
            for j in adj[i]:
                if visited[j]:
                    continue
                visited[j] = True
                if match_right[j] == -1 or dfs(match_right[j], visited):
                    match_right[j] = i
                    return True
            return False
        
        for i in range(1, N + 1):
            visited = [False] * (L + 1)
            if not dfs(i, visited):
                return False
        return True
    
    def compute_f(r):
        # Precompute div_masks for this r
        div_masks = [0] * 145
        for j in range(1, 145):
            x = r + j
            mask = 0
            for i in range(1, N + 1):
                if x % i == 0:
                    mask |= (1 << (i - 1))
            div_masks[j] = mask
        
        # Find least L
        for L in range(1, 145):
            if has_matching_from_masks(div_masks, L):
                return L
        return 144
    
    best_r = 0
    best_f = 0
    
    for r in range(M):
        if time.time() > DEADLINE:
            break
        f_val = compute_f(r)
        if f_val > best_f:
            best_f = f_val
            best_r = r
    
    return int(best_r)

# EVOLVE-BLOCK-END
