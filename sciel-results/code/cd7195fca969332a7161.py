"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    full_mask = (1 << n) - 1
    
    # Precompute is_clique and is_ind for all subsets
    # is_clique[mask]: True if mask induces a clique
    # is_ind[mask]: True if mask induces an independent set
    is_clique = [False] * (1 << n)
    is_ind = [False] * (1 << n)
    is_clique[0] = True
    is_ind[0] = True
    
    for mask in range(1, 1 << n):
        # Get least significant bit
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # Check clique: v must be connected to all in rest, and rest must be clique
        if is_clique[rest] and (adj[v] & rest) == rest:
            is_clique[mask] = True
            
        # Check independent set: v must not be connected to any in rest, and rest must be independent
        if is_ind[rest] and (adj[v] & rest) == 0:
            is_ind[mask] = True
            
    is_valid = [is_clique[m] or is_ind[m] for m in range(1 << n)]
    
    # Memoization for the search: min_parts[rem] = minimum number of parts needed to cover 'rem'
    INF = n + 1
    min_parts = [INF] * (1 << n)
    min_parts[0] = 0
    
    # Also store the actual partition for reconstruction
    # solution[rem] = (first_part_mask, remaining_mask) where first_part_mask is a valid subset of rem
    # and remaining_mask = rem ^ first_part_mask
    solution = [None] * (1 << n)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0
    
    # We compute min_parts[rem] for all rem in increasing order of popcount?
    # No, the recurrence is min_parts[rem] = 1 + min(min_parts[rem ^ sub] for sub in valid_subsets(rem))
    # This is not strictly increasing in rem value, but we can use memoized recursion or iterative DP.
    # Since n=14, 2^14 = 16384 states.
    # For each state rem, we iterate all submasks sub of rem.
    # Total complexity is sum_{rem} 2^{popcount(rem)} = 3^n.
    # 3^14 = 4,782,969. This is very fast.
    
    # Iterative DP
    # We need to iterate rem such that we have computed min_parts[rem ^ sub] before min_parts[rem].
    # Since rem ^ sub is always a proper subset of rem (sub is non-empty), popcount(rem ^ sub) < popcount(rem).
    # So we can iterate by increasing popcount.
    
    # Group masks by popcount
    masks_by_popcount = [[] for _ in range(n + 1)]
    for mask in range(1, 1 << n):
        pc = mask.bit_count()
        masks_by_popcount[pc].append(mask)
        
    for pc in range(1, n + 1):
        for rem in masks_by_popcount[pc]:
            if time.time() > deadline:
                # Fallback to greedy
                break
            # Find best submask
            best_val = INF
            best_sub = None
            
            # Iterate all non-empty submasks of rem
            sub = rem
            while sub:
                if is_valid[sub]:
                    val = min_parts[rem ^ sub] + 1
                    if val < best_val:
                        best_val = val
                        best_sub = sub
                sub = (sub - 1) & rem
                
            min_parts[rem] = best_val
            if best_sub is not None:
                solution[rem] = (best_sub, rem ^ best_sub)
                
        if time.time() > deadline:
            break
    
    # Reconstruct solution
    if min_parts[full_mask] < INF:
        parts = []
        rem = full_mask
        while rem:
            sub, rem = solution[rem]
            # Convert sub mask to list of vertices
            part_verts = []
            temp_sub = sub
            while temp_sub:
                b = temp_sub & -temp_sub
                idx = b.bit_length() - 1
                part_verts.append(idx)
                temp_sub ^= b
            parts.append(part_verts)
            
        # Verify and return
        all_verts = []
        for part in parts:
            all_verts.extend(part)
        if sorted(all_verts) != list(range(n)):
            # Should not happen, but fallback
            pass
        else:
            # Check validity
            valid = True
            for part in parts:
                mask = 0
                for v in part:
                    mask |= (1 << v)
                if not is_valid[mask]:
                    valid = False
                    break
            if valid:
                return parts
    
    # Fallback greedy solution
    es = {tuple(e) for e in edges}
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
# EVOLVE-BLOCK-END
