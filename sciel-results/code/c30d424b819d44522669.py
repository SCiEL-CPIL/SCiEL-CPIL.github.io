import time
from functools import lru_cache

def _kuhn(adj, n_left, n_right):
    """Kuhn's algorithm for bipartite matching. adj[i] is list of right nodes."""
    match_r = [-1] * n_right
    # Order by degree (fewest options first)
    order = sorted(range(n_left), key=lambda i: len(adj[i]))
    
    def dfs(u, visited):
        for v in adj[u]:
            if v in visited:
                continue
            visited.add(v)
            if match_r[v] == -1 or dfs(match_r[v], visited):
                match_r[v] = u
                return True
        return False
    
    for u in order:
        if not dfs(u, set()):
            return False
    return True

def _f_for_m(m):
    """Compute f(12, m) = least L such that (m, m+L] has distinct b_i with i|b_i."""
    n = 12
    # Theoretical lower bound: max distance to first multiple
    min_L = 0
    for i in range(1, n + 1):
        d = i - (m % i)
        if d == 0:
            d = i
        if d > min_L:
            min_L = d
    
    # Upper bound: 144 is safe
    lo, hi = min_L, 144
    best = 144
    
    while lo <= hi:
        mid = (lo + hi) // 2
        L = mid
        # Build adjacency: for each i, which positions j in 0..L-1 have (m+1+j) divisible by i
        adj = []
        feasible = True
        for i in range(1, n + 1):
            opts = []
            # (m+1+j) % i == 0 => j ≡ (-m-1) % i, j in [0, L-1]
            r = (-m - 1) % i
            for j in range(r, L, i):
                opts.append(j)
            if not opts:
                feasible = False
                break
            adj.append(opts)
        
        if not feasible:
            lo = mid + 1
            continue
        
        if _kuhn(adj, n, L):
            best = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    return best

def construct():
    t0 = time.time()
    deadline = 80.0
    n = 12
    period = 27720
    
    best_m = 0
    best_L = 0
    
    # Phase 1: Quick scan to find candidates with high lower bound
    # Pre-compute lower bound for each residue
    # lb[m] = max over i of distance to first multiple of i after m
    # Then only do full matching check for residues where lb is high
    
    # But actually, let's just do the full scan with time guard
    # 27720 iterations * ~8 binary search steps * matching cost
    
    # Optimize: precompute the structure
    for m in range(period):
        if time.time() - t0 > deadline:
            break
        
        L = _f_for_m(m)
        if L > best_L:
            best_L = L
            best_m = m
    
    return int(best_m)

# EVOLVE-BLOCK-START
# EVOLVE-BLOCK-END
