"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    deadline = time.time() + 55  # safety margin before 60s limit
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # non_adj[v] = bitmask of vertices NOT adjacent to v (excluding v itself)
    non_adj = [0] * n
    for v in range(n):
        non_adj[v] = 0
        for u in range(n):
            if u != v and not (adj[v] & (1 << u)):
                non_adj[v] |= (1 << u)
    
    full_mask = (1 << n) - 1
    
    # Precompute: for each subset, is it a clique? is it independent?
    # is_clique[S] = True if S induces a clique
    # is_indep[S] = True if S induces an independent set
    is_clique = [False] * (1 << n)
    is_indep = [False] * (1 << n)
    
    is_clique[0] = True
    is_indep[0] = True
    
    for S in range(1, 1 << n):
        # Get lowest set bit
        lb = S & (-S)
        v = lb.bit_length() - 1
        rest = S ^ lb
        
        # S is a clique iff rest is a clique AND v is adjacent to all in rest
        if is_clique[rest] and (rest & ~adj[v]) == 0:
            is_clique[S] = True
        
        # S is independent iff rest is independent AND v is not adjacent to any in rest
        if is_indep[rest] and (rest & adj[v]) == 0:
            is_indep[S] = True
    
    # A subset is "valid" if it's a clique or independent set
    is_valid = [is_clique[S] or is_indep[S] for S in range(1 << n)]
    
    # DP: dp[S] = min number of valid parts to cover S
    # dp[0] = 0
    # dp[S] = min over valid T ⊆ S of (dp[S ^ T] + 1)
    INF = 10**9
    dp = [INF] * (1 << n)
    dp[0] = 0
    
    for S in range(1, 1 << n):
        # Iterate over all valid submasks T of S
        T = S
        while T:
            if is_valid[T]:
                prev = S ^ T
                if dp[prev] + 1 < dp[S]:
                    dp[S] = dp[prev] + 1
            T = (T - 1) & S
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    if time.time() > deadline or dp[full_mask] == INF:
        # Fallback to greedy
        es = {tuple(e) for e in P['edges']}
        parts = []
        for v in range(n):
            for part in parts:
                trial = part + [v]
                hits = [(u, w) in es for i, u in enumerate(trial) for w in trial[i+1:]]
                if all(hits) or not any(hits):
                    part.append(v)
                    break
            else:
                parts.append([v])
        return parts
    
    # Reconstruct the partition from dp
    # We need to track which T was chosen at each step
    # Redo DP with parent tracking
    parent = [-1] * (1 << n)  # parent[S] = T (the valid subset chosen)
    dp2 = [INF] * (1 << n)
    dp2[0] = 0
    
    for S in range(1, 1 << n):
        T = S
        while T:
            if is_valid[T]:
                prev = S ^ T
                if dp2[prev] + 1 < dp2[S]:
                    dp2[S] = dp2[prev] + 1
                    parent[S] = T
            T = (T - 1) & S
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    if time.time() > deadline or dp2[full_mask] == INF:
        es = {tuple(e) for e in P['edges']}
        parts = []
        for v in range(n):
            for part in parts:
                trial = part + [v]
                hits = [(u, w) in es for i, u in enumerate(trial) for w in trial[i+1:]]
                if all(hits) or not any(hits):
                    part.append(v)
                    break
            else:
                parts.append([v])
        return parts
    
    # Reconstruct
    parts = []
    S = full_mask
    while S:
        T = parent[S]
        if T == -1:
            # Shouldn't happen, but fallback
            break
        part = []
        for v in range(n):
            if T & (1 << v):
                part.append(v)
        parts.append(part)
        S ^= T
    
    # Convert to list of lists of Python ints
    result = []
    for part in parts:
        result.append([int(v) for v in part])
    
    return result

# EVOLVE-BLOCK-END
