"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s deadline, 15s safety margin
    
    n = P['n']
    edges_set = set()
    for e in P['edges']:
        edges_set.add((e[0], e[1]))
        edges_set.add((e[1], e[0]))
    
    # Precompute adjacency as bitmasks
    adj = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j and (i, j) in edges_set:
                adj[i] |= (1 << j)
    
    # Precompute for each subset whether it's a clique or independent set
    # Use bitmask operations for efficiency
    valid = [False] * (1 << n)
    full_mask = (1 << n) - 1
    
    for mask in range(1, 1 << n):
        # Check if mask is a clique: for all u in mask, (mask & ~(1<<u)) must be subset of adj[u]
        is_clique = True
        m = mask
        while m:
            lsb = m & (-m)
            u = lsb.bit_length() - 1
            if (mask & ~lsb) & ~adj[u]:
                is_clique = False
                break
            m ^= lsb
        
        if not is_clique:
            # Check if mask is an independent set: for all u in mask, (mask & ~(1<<u)) must be subset of ~adj[u]
            is_indep = True
            m = mask
            while m:
                lsb = m & (-m)
                u = lsb.bit_length() - 1
                if (mask & ~lsb) & adj[u]:
                    is_indep = False
                    break
                m ^= lsb
            if is_indep:
                valid[mask] = True
        else:
            valid[mask] = True
    
    # Check deadline
    if time.time() > deadline:
        return [[i] for i in range(n)]
    
    # DP: dp[mask] = minimum number of valid parts to cover vertices in mask
    INF = 256  # n+1 is enough
    dp = [INF] * (1 << n)
    dp[0] = 0
    
    # Also track which subset was used for reconstruction
    parent = [-1] * (1 << n)
    
    for mask in range(1, 1 << n):
        # Try all valid subsets S that are subsets of mask
        sub = mask
        while sub > 0:
            if time.time() > deadline:
                break
            if valid[sub]:
                prev = mask ^ sub
                if dp[prev] + 1 < dp[mask]:
                    dp[mask] = dp[prev] + 1
                    parent[mask] = sub
            sub = (sub - 1) & mask
    
    # Reconstruct the partition
    parts = []
    mask = full_mask
    while mask > 0:
        sub = parent[mask]
        if sub == -1:
            part = [i for i in range(n) if mask & (1 << i)]
            parts.append(part)
            break
        part = []
        for i in range(n):
            if sub & (1 << i):
                part.append(i)
        parts.append(part)
        mask ^= sub
    
    # Convert to list of lists of Python ints
    result = [list(p) for p in parts]
    return result

# EVOLVE-BLOCK-END
