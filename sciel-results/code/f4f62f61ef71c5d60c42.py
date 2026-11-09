import time
from functools import lru_cache

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    full_mask = (1 << n) - 1
    start_time = time.time()
    deadline = start_time + 80
    
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Precompute validity
    is_valid = [False] * (1 << n)
    is_valid[0] = True
    for mask in range(1, 1 << n):
        is_clique = True
        is_indep = True
        m = mask
        while m:
            lsb = m & -m
            u = lsb.bit_length() - 1
            rest = mask ^ lsb
            if rest & ~adj[u]:
                is_clique = False
            if rest & adj[u]:
                is_indep = False
            if not is_clique and not is_indep:
                break
            m ^= lsb
        is_valid[mask] = is_clique or is_indep
    
    # Memoized recursion
    memo = {}
    
    def solve(mask):
        if mask == 0:
            return 0
        if time.time() > deadline:
            return 10**9
        if mask in memo:
            return memo[mask]
        
        best = 10**9
        sub = mask
        while sub:
            if is_valid[sub]:
                val = solve(mask ^ sub) + 1
                if val < best:
                    best = val
            sub = (sub - 1) & mask
        
        memo[mask] = best
        return best
    
    result = solve(full_mask)
    
    if result >= 10**9:
        # Fallback: greedy
        remaining = full_mask
        parts = []
        while remaining:
            best = 0
            best_size = 0
            sub = remaining
            while sub:
                if is_valid[sub]:
                    sz = sub.bit_count()
                    if sz > best_size:
                        best_size = sz
                        best = sub
                sub = (sub - 1) & remaining
            if best == 0:
                lsb = remaining & -remaining
                best = lsb
            m = best
            part_vertices = []
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                part_vertices.append(v)
                m ^= lsb
            parts.append(part_vertices)
            remaining ^= best
        return parts
    
    # Reconstruct
    # Need to track which sub was chosen
    memo2 = {}
    
    def solve2(mask):
        if mask == 0:
            return 0, 0
        if time.time() > deadline:
            return 10**9, 0
        if mask in memo2:
            return memo2[mask]
        
        best = 10**9
        best_sub = 0
        sub = mask
        while sub:
            if is_valid[sub]:
                val, _ = solve2(mask ^ sub)
                if val + 1 < best:
                    best = val + 1
                    best_sub = sub
            sub = (sub - 1) & mask
        
        memo2[mask] = (best, best_sub)
        return best, best_sub
    
    total, _ = solve2(full_mask)
    
    parts = []
    current = full_mask
    while current:
        _, sub = memo2[current]
        m = sub
        part_vertices = []
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            part_vertices.append(v)
            m ^= lsb
        parts.append(part_vertices)
        current ^= sub
    
    return parts

