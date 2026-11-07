import time
import functools

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency as bitmasks for fast operations
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    # Precompute: for each mask, whether it's a clique or independent set
    # Use bit operations: a set S is a clique iff for all v in S, S \ {v} subset of adj[v]
    # A set S is independent iff for all v in S, S \ {v} disjoint from adj[v]
    
    full_mask = (1 << n) - 1
    
    # is_clique[mask] and is_indep[mask]
    is_clique = [False] * (1 << n)
    is_indep = [False] * (1 << n)
    is_clique[0] = True
    is_indep[0] = True
    
    for mask in range(1, 1 << n):
        # Get lowest set bit
        lsb = mask & (-mask)
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # Check if adding v to 'rest' keeps it a clique:
        # rest must be a clique AND all vertices in rest must be adjacent to v
        if is_clique[rest] and (rest & ~adj_mask[v]) == 0:
            is_clique[mask] = True
        
        # Check if adding v to 'rest' keeps it independent:
        # rest must be independent AND no vertex in rest is adjacent to v
        if is_indep[rest] and (rest & adj_mask[v]) == 0:
            is_indep[mask] = True
    
    # valid[mask] = is_clique[mask] or is_indep[mask]
    valid = [False] * (1 << n)
    for mask in range(1, 1 << n):
        valid[mask] = is_clique[mask] or is_indep[mask]
    
    # For each vertex, precompute list of valid masks containing it, sorted by size descending
    masks_by_vertex = [[] for _ in range(n)]
    for mask in range(1, 1 << n):
        if valid[mask]:
            m = mask
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                masks_by_vertex[v].append(mask)
                m ^= lsb
    
    # Sort by popcount descending (larger parts first for faster search)
    for v in range(n):
        masks_by_vertex[v].sort(key=lambda m: m.bit_count(), reverse=True)
    
    # Precompute popcount for all masks
    popcount = [0] * (1 << n)
    for mask in range(1, 1 << n):
        popcount[mask] = popcount[mask >> 1] + (mask & 1)
    
    # Precompute max_valid_part_size_in[mask]: the size of the largest valid part that is a subset of mask
    # This is expensive to compute for all masks, so we'll compute it on-the-fly with memoization
    # Actually, for pruning, we can use: if remaining has r vertices, and the largest valid part 
    # subset of remaining has size s, then we need at least ceil(r/s) parts.
    # But computing this exactly is as hard as the original problem.
    
    # Simpler pruning: lower bound on parts = ceil(remaining_count / max_possible_part_size)
    # where max_possible_part_size = max over all valid parts of popcount
    max_part_size = max(popcount[m] for m in range(1, 1 << n) if valid[m])
    
    # Backtracking search for minimum k
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # Try k = 1, 2, ..., n
    for k in range(1, n + 1):
        if time.time() > deadline:
            break
        
        # Check if k parts can cover all vertices
        # Use memoized DFS
        memo = {}
        
        def can_cover(remaining, k_left):
            if time.time() > deadline:
                return False
            if k_left == 0:
                return remaining == 0
            if remaining == 0:
                return False  # parts must be non-empty
            
            key = (remaining, k_left)
            if key in memo:
                return memo[key]
            
            # Lower bound: need at least ceil(popcount(remaining) / max_part_size) parts
            r = popcount[remaining]
            if r > k_left * max_part_size:
                memo[key] = False
                return False
            
            # Pick vertex with fewest valid extensions (most constrained)
            # Try each vertex in remaining, count valid masks containing it that are subset of remaining
            best_v = -1
            best_count = float('inf')
            m = remaining
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                count = 0
                for pm in masks_by_vertex[v]:
                    if (pm & remaining) == pm:
                        count += 1
                        if count >= best_count:
                            break
                if count < best_count:
                    best_count = count
                    best_v = v
                    if count == 0:
                        break
                m ^= lsb
            
            if best_v == -1 or best_count == 0:
                memo[key] = False
                return False
            
            # Try all valid masks containing best_v that are subsets of remaining
            for pm in masks_by_vertex[best_v]:
                if (pm & remaining) == pm:
                    if can_cover(remaining ^ pm, k_left - 1):
                        memo[key] = True
                        return True
            
            memo[key] = False
            return False
        
        if can_cover(full_mask, k):
            # Found k. Reconstruct the partition.
            parts = []
            remaining = full_mask
            for _ in range(k):
                # Find which part was chosen
                # Re-run the search but record choices
                # Since we know it's feasible, do a fresh DFS recording path
                path = []
                
                def find_path(rem, kl):
                    if time.time() > deadline:
                        return False
                    if kl == 0:
                        return rem == 0
                    if rem == 0:
                        return False
                    
                    # Pick most constrained vertex
                    best_v2 = -1
                    best_count2 = float('inf')
                    m2 = rem
                    while m2:
                        lsb2 = m2 & (-m2)
                        v2 = lsb2.bit_length() - 1
                        count2 = 0
                        for pm2 in masks_by_vertex[v2]:
                            if (pm2 & rem) == pm2:
                                count2 += 1
                                if count2 >= best_count2:
                                    break
                        if count2 < best_count2:
                            best_count2 = count2
                            best_v2 = v2
                            if count2 == 0:
                                break
                        m2 ^= lsb2
                    
                    if best_v2 == -1:
                        return False
                    
                    for pm2 in masks_by_vertex[best_v2]:
                        if (pm2 & rem) == pm2:
                            path.append(pm2)
                            if find_path(rem ^ pm2, kl - 1):
                                return True
                            path.pop()
                    return False
                
                if find_path(remaining, k):
                    # Extract the last part chosen (the one for the first vertex)
                    # Actually, path contains all k masks in order
                    pass
                
                # The path should have k elements. Take the first one.
                if path:
                    part_mask = path[0]
                    # Convert to list of vertices
                    verts = []
                    mm = part_mask
                    while mm:
                        lsb_m = mm & (-mm)
                        v_m = lsb_m.bit_length() - 1
                        verts.append(v_m)
                        mm ^= lsb_m
                    parts.append(verts)
                    remaining ^= part_mask
                    path = []  # Reset for next part
                else:
                    # Should not happen
                    break
            
            # Verify and return
            if len(parts) == k and all(len(p) > 0 for p in parts):
                # Check all vertices covered exactly once
                all_verts = []
                for p in parts:
                    all_verts.extend(p)
                if sorted(all_verts) == list(range(n)):
                    # Ensure types are correct
                    return [[int(v) for v in p] for p in parts]
    
    # Fallback: greedy
    es = {tuple(e) for e in edges}
    parts = []
    for v in range(n):
        placed = False
        for part in parts:
            trial = part + [v]
            hits = [(u, w) in es for i, u in enumerate(trial) for w in trial[i+1:]]
            if all(hits) or not any(hits):
                part.append(v)
                placed = True
                break
        if not placed:
            parts.append([v])
    return parts
