"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s deadline, 10s safety margin
    
    n = P['n']
    edges = set()
    for e in P['edges']:
        edges.add((e[0], e[1]))
        edges.add((e[1], e[0]))
    
    # Precompute adjacency as bitmasks
    adj = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j and (i, j) in edges:
                adj[i] |= (1 << j)
    
    # Precompute for each subset whether it's a clique or independent set
    # valid[mask] = True if mask induces a clique or an independent set
    valid = [False] * (1 << n)
    
    # Optimization: Precompute clique and independent set properties using DP on masks
    # is_clique[mask] = is_clique[mask without lowest bit] AND (lowest bit connected to all others in mask)
    # is_indep[mask] = is_indep[mask without lowest bit] AND (lowest bit NOT connected to any others in mask)
    
    is_clique = [False] * (1 << n)
    is_indep = [False] * (1 << n)
    is_clique[0] = True
    is_indep[0] = True
    
    for mask in range(1, 1 << n):
        # Get lowest set bit
        lsb = mask & -mask
        v = (lsb.bit_length() - 1)
        rest = mask ^ lsb
        
        # Check clique
        if is_clique[rest]:
            # v must be connected to all vertices in rest
            # adj[v] has bits for neighbors.
            # rest has bits for vertices in rest.
            # We need (rest & ~adj[v]) == 0
            if (rest & ~adj[v]) == 0:
                is_clique[mask] = True
        
        # Check independent set
        if is_indep[rest]:
            # v must NOT be connected to any vertex in rest
            # We need (rest & adj[v]) == 0
            if (rest & adj[v]) == 0:
                is_indep[mask] = True
        
        if is_clique[mask] or is_indep[mask]:
            valid[mask] = True
            
    # Check deadline
    if time.time() > deadline:
        # Fallback: return each vertex as its own part
        return [[i] for i in range(n)]
    
    # DP: dp[mask] = minimum number of valid parts to cover vertices in mask
    INF = float('inf')
    dp = [INF] * (1 << n)
    dp[0] = 0
    
    # Also track which subset was used for reconstruction
    parent = [-1] * (1 << n)
    
    for mask in range(1, 1 << n):
        # Try all valid subsets S that are subsets of mask
        # Optimization: iterate only over submasks of mask
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
            
    full_mask = (1 << n) - 1
    min_parts = dp[full_mask]
    
    # Reconstruct the partition
    parts = []
    mask = full_mask
    while mask > 0:
        sub = parent[mask]
        if sub == -1:
            # Should not happen, but fallback
            part = [i for i in range(n) if mask & (1 << i)]
            parts.append(part)
            break
        part = []
        for i in range(n):
            if sub & (1 << i):
                part.append(i)
        parts.append(part)
        mask ^= sub
        
    # Convert to list of lists of ints
    result = [list(p) for p in parts]
    
    # Final type check and conversion
    final_result = []
    for part in result:
        final_part = []
        for v in part:
            if isinstance(v, int) and not isinstance(v, bool):
                final_part.append(v)
        final_result.append(final_part)
        
    return final_result

# EVOLVE-BLOCK-END
