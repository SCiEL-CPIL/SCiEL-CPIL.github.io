"""Cochromatic partition via optimized subset DP."""
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    start_time = time.time()
    deadline = 80.0  # safety margin before 90s kill
    
    n = P['n']
    full_mask = (1 << n) - 1
    size = 1 << n
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Precompute is_indep and is_clique for all masks
    is_indep = [False] * size
    is_clique = [False] * size
    is_indep[0] = True
    is_clique[0] = True
    
    for mask in range(1, size):
        lb = mask & (-mask)
        v = lb.bit_length() - 1
        rest = mask ^ lb
        is_indep[mask] = is_indep[rest] and (adj[v] & rest) == 0
        is_clique[mask] = is_clique[rest] and (adj[v] & rest) == rest
    
    # Identify all valid parts (non-empty cliques or independent sets)
    is_valid = [False] * size
    valid_masks = []
    for mask in range(1, size):
        if is_indep[mask] or is_clique[mask]:
            is_valid[mask] = True
            valid_masks.append(mask)
    
    # DP: dp[mask] = min number of parts to partition vertices in mask
    INF = n + 1
    dp = [INF] * size
    parent = [0] * size
    dp[0] = 0
    
    # Sort valid_masks? Not strictly necessary, but might help if we want to break early?
    # We just need the min.
    
    for mask in range(1, size):
        # Early check: if mask itself is a valid part
        if is_valid[mask]:
            dp[mask] = 1
            parent[mask] = mask
            continue
        
        # Iterate over valid submasks
        # Optimization: Only check valid_masks that are submasks of mask
        best_val = INF
        best_sub = 0
        
        # Check deadline periodically? 
        # The loop over valid_masks is small, so checking every iteration is overhead.
        # But if valid_masks is large, we might need it.
        # For n=14, valid_masks is likely < 2000.
        
        for sub in valid_masks:
            # Check if sub is a submask of mask
            if (sub & ~mask) == 0:
                # sub is a valid part and submask of mask
                # Since mask is not valid, sub != mask (if sub==mask, is_valid[mask] would be True)
                rest = mask ^ sub
                val = dp[rest] + 1
                if val < best_val:
                    best_val = val
                    best_sub = sub
                    if best_val == 2:
                        break
        
        dp[mask] = best_val
        parent[mask] = best_sub
        
        # Deadline check
        if time.time() - start_time > deadline:
            # Fallback: reconstruct best effort
            parts = []
            current = full_mask
            while current:
                if parent[current]:
                    part_mask = parent[current]
                    verts = [i for i in range(n) if (part_mask >> i) & 1]
                    parts.append(verts)
                    current ^= part_mask
                else:
                    # Fallback: single vertex
                    lb = current & (-current)
                    v = lb.bit_length() - 1
                    parts.append([v])
                    current ^= lb
            return [[int(v) for v in part] for part in parts]
    
    # Reconstruct the optimal partition
    parts = []
    current = full_mask
    while current:
        part_mask = parent[current]
        verts = [i for i in range(n) if (part_mask >> i) & 1]
        parts.append(verts)
        current ^= part_mask
    
    # Ensure types: list of lists of Python ints
    return [[int(v) for v in part] for part in parts]
