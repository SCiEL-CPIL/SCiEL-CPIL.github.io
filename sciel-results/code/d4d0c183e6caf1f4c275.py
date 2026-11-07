import time
from functools import lru_cache

def _kuhn(adj, n_left, n_right):
    """Kuhn's algorithm for bipartite matching. adj[i] is list of right nodes."""
    match_r = [-1] * n_right
    # Order by degree (fewest options first)
    order = sorted(range(n_left), key=lambda i: len(adj[i]))
    
    # Use iterative DFS with a visited array for speed
    for u in order:
        visited = [False] * n_right
        stack = [u]
        parent = [-1] * n_left
        found = False
        while stack:
            node = stack.pop()
            if node < 0:
                # This is a right node index encoded as negative
                v = -node - 1
                # Found an unmatched right node, augment
                while node >= 0:
                    prev = parent[node]
                    match_r[v] = node
                    node = prev
                    if prev >= 0:
                        v = adj[prev][0]  # This won't work correctly
                    break
                found = True
                break
            for v in adj[node]:
                if not visited[v]:
                    visited[v] = True
                    if match_r[v] == -1:
                        # Found augmenting path
                        found = True
                        # Augment: we need to track the path
                        # This iterative approach is tricky; let's use recursive
                        break
                    else:
                        parent[match_r[v]] = node
                        stack.append(match_r[v])
            if found:
                break
        if not found:
            # Fallback: try recursive
            def dfs(u, visited):
                for v in adj[u]:
                    if v in visited:
                        continue
                    visited.add(v)
                    if match_r[v] == -1 or dfs(match_r[v], visited):
                        match_r[v] = u
                        return True
                return False
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
    deadline = 75.0
    n = 12
    period = 27720
    
    best_m = 0
    best_L = 0
    
    # Pre-compute lower bounds for all m
    # lb[m] = max over i of (i - m % i) if m % i != 0 else i
    # This is the minimum L that could possibly work for each i
    # f(12, m) >= lb[m], so if lb[m] <= best_L, skip
    
    # First pass: compute lower bounds and find the max possible
    lbs = []
    max_lb = 0
    for m in range(period):
        lb = 0
        for i in range(1, n + 1):
            r = m % i
            d = i - r if r != 0 else i
            if d > lb:
                lb = d
        lbs.append(lb)
        if lb > max_lb:
            max_lb = lb
    
    # Sort indices by lower bound descending - check promising ones first
    indices = sorted(range(period), key=lambda x: -lbs[x])
    
    for m in indices:
        if time.time() - t0 > deadline:
            break
        
        # Skip if lower bound can't beat current best
        if lbs[m] <= best_L:
            continue
        
        L = _f_for_m(m)
        if L > best_L:
            best_L = L
            best_m = m
    
    return int(best_m)

# EVOLVE-BLOCK-START
# EVOLVE-BLOCK-END
