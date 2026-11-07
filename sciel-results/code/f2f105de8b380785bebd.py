"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    deadline = time.time() + 80  # safety margin before 90s limit
    
    # Build adjacency bitmask for each vertex
    adj_mask = [0] * n
    for u, v in P['edges']:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    # Precompute for each subset: is it a clique? is it an independent set?
    full_mask = (1 << n) - 1
    
    is_clique = [False] * (1 << n)
    is_ind = [False] * (1 << n)
    
    # Single vertices: both clique and independent set
    for v in range(n):
        is_clique[1 << v] = True
        is_ind[1 << v] = True
    
    # Build up using the lowest set bit
    for mask in range(1, 1 << n):
        lsb = mask & (-mask)
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        if rest == 0:
            continue
        
        # Check clique: v must be adjacent to all vertices in rest
        if is_clique[rest]:
            # v is adjacent to all in rest iff (rest & ~adj_mask[v]) == 0
            if (rest & ~adj_mask[v]) == 0:
                is_clique[mask] = True
        
        # Check independent set: v must be non-adjacent to all vertices in rest
        if is_ind[rest]:
            # v is non-adjacent to all in rest iff (rest & adj_mask[v]) == 0
            if (rest & adj_mask[v]) == 0:
                is_ind[mask] = True
    
    # DP: dp[mask] = minimum number of parts to cover vertices in mask
    INF = n + 1
    dp = [INF] * (1 << n)
    dp[0] = 0
    parent = [0] * (1 << n)
    
    # Precompute list of valid subsets (cliques or independent sets) for faster lookup
    # Actually, we'll just check is_clique[sub] or is_ind[sub] directly
    
    # Iterate over all masks in increasing order
    for mask in range(1, 1 << n):
        if time.time() > deadline:
            break
        sub = mask
        best = INF
        best_sub = 0
        while sub > 0:
            if is_clique[sub] or is_ind[sub]:
                val = dp[mask ^ sub] + 1
                if val < best:
                    best = val
                    best_sub = sub
                    if best == 1:
                        break
            sub = (sub - 1) & mask
        
        dp[mask] = best
        parent[mask] = best_sub
    
    # Reconstruct the partition
    partition = []
    mask = full_mask
    while mask > 0:
        if time.time() > deadline:
            break
        sub = parent[mask]
        if sub == 0:
            # Fallback: add remaining as singletons
            while mask > 0:
                lsb = mask & (-mask)
                v = lsb.bit_length() - 1
                partition.append([v])
                mask ^= lsb
            break
        # Extract vertices from sub
        verts = []
        m = sub
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            verts.append(v)
            m ^= lsb
        partition.append(verts)
        mask ^= sub
    
    # Ensure all vertices are covered exactly once
    covered = [0] * n
    for part in partition:
        for v in part:
            if 0 <= v < n:
                covered[v] += 1
    
    # If something went wrong, fall back to greedy
    if any(c != 1 for c in covered):
        # Greedy fallback
        es = set()
        for e in P['edges']:
            es.add((e[0], e[1]))
            es.add((e[1], e[0]))
        parts = []
        for v in range(n):
            placed = False
            for part in parts:
                # Check if adding v keeps part as clique or independent set
                trial = part + [v]
                all_edges = True
                no_edges = True
                for i in range(len(trial)):
                    for j in range(i + 1, len(trial)):
                        if (trial[i], trial[j]) in es:
                            no_edges = False
                        else:
                            all_edges = False
                if all_edges or no_edges:
                    part.append(v)
                    placed = True
                    break
            if not placed:
                parts.append([v])
        return parts
    
    # Convert to pure Python types
    result = []
    for part in partition:
        result.append([int(v) for v in part])
    
    return result

# EVOLVE-BLOCK-END
