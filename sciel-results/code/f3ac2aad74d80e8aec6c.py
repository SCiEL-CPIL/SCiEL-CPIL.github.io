import math
import sys

def _compute_f_fast(m, n=12):
    """Compute f(n, m) using optimized matching."""
    # Lower bound for L
    # First multiple of i in (m, ...] is m + 1 + ((-m - 1) % i)
    # Let offset_i = 1 + ((-m - 1) % i)
    # L must be >= max(offset_i)
    offsets = [0] * (n + 1)
    for i in range(1, n + 1):
        offsets[i] = 1 + ((-m - 1) % i)
    
    L_min = max(offsets)
    
    # Upper bound is known to be small, e.g., 144
    for L in range(L_min, 145):
        # Build adjacency
        # adj[i] = list of j in 0..L-1 such that (m+1+j) % i == 0
        # (m+1+j) % i == 0 => j % i == (-m-1) % i
        # j = rem, rem+i, ...
        adj = [None] * (n + 1)
        for i in range(1, n + 1):
            rem = (-m - 1) % i
            if rem >= L:
                adj[i] = []
            else:
                # Generate list
                # j = rem, rem+i, ... < L
                adj[i] = list(range(rem, L, i))
        
        # Check matching
        match_right = [-1] * L
        match_left = [-1] * (n + 1)
        
        # DFS
        # To speed up, we can order i by number of candidates (smallest first)
        # This often helps DFS find matches faster
        order = sorted(range(1, n + 1), key=lambda x: len(adj[x]))
        
        def dfs(i, visited):
            for j in adj[i]:
                if visited[j]:
                    continue
                visited[j] = True
                if match_right[j] == -1 or dfs(match_right[j], visited):
                    match_right[j] = i
                    match_left[i] = j
                    return True
            return False
        
        ok = True
        for i in order:
            if not adj[i]:
                ok = False
                break
            visited = [False] * L
            if not dfs(i, visited):
                ok = False
                break
        
        if ok:
            return L
            
    return 144 # Fallback, shouldn't happen

def construct():
    """Find m in [0, 1000000] that maximizes f(12, m)."""
    # Period is LCM(1..12)
    lcm_1_12 = 27720
    
    best_f = 0
    best_r = 0
    
    # We need to check all residues r in [0, lcm_1_12)
    # To speed this up, we can note that f(12, m) depends on m mod lcm.
    # We iterate r from 0 to lcm_1_12 - 1.
    
    # Optimization: 
    # The computation of _compute_f_fast is the bottleneck.
    # 27720 calls.
    # Each call does matching.
    # Let's hope it's fast enough.
    
    for r in range(lcm_1_12):
        # Compute f for residue r
        # Note: f(12, r) is the same as f(12, r + k*lcm)
        f_val = _compute_f_fast(r, 12)
        
        if f_val > best_f:
            best_f = f_val
            best_r = r
            
    # Find the smallest m in [0, 1000000] such that m % lcm_1_12 == best_r
    # m = best_r + k * lcm_1_12
    # Smallest non-negative m is best_r (if best_r <= 1000000)
    # Since lcm is 27720 and best_r < 27720, best_r is always <= 1000000.
    m_val = best_r
    
    # Ensure m_val is in range
    if m_val < 0 or m_val > 1000000:
         # Find k such that 0 <= best_r + k*lcm <= 1000000
         # k = 0 works for best_r < 1000000
         # If best_r was large, we might need to adjust, but best_r < 27720
         pass
         
    return m_val
