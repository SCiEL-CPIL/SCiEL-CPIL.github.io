import time
from math import ceil

def construct():
    P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    full_mask = (1 << n) - 1
    N = 1 << n
    
    # Precompute popcount for all masks
    popcount = [0] * N
    for m in range(1, N):
        popcount[m] = popcount[m >> 1] + (m & 1)
    
    # Precompute is_clique and is_ind for all subsets
    is_clique = [False] * N
    is_ind = [False] * N
    is_clique[0] = True
    is_ind[0] = True
    
    for mask in range(1, N):
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        if is_clique[rest] and (adj[v] & rest) == rest:
            is_clique[mask] = True
            
        if is_ind[rest] and (adj[v] & rest) == 0:
            is_ind[mask] = True
            
    is_valid = [is_clique[m] or is_ind[m] for m in range(N)]
    
    # Precompute omega[mask] = size of max clique in subgraph induced by mask
    # alpha[mask] = size of max independent set in subgraph induced by mask
    omega = [0] * N
    alpha = [0] * N
    
    for mask in range(1, N):
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # Max clique: either v is in max clique (then rest must be clique-compatible) or not
        # omega[mask] = max(omega[rest], 1 + omega[rest & adj[v]])
        omega[mask] = max(omega[rest], 1 + omega[rest & adj[v]])
        
        # Max independent set: either v is in max ind set (then rest must be non-neighbor-compatible) or not
        # alpha[mask] = max(alpha[rest], 1 + alpha[rest & ~adj[v] & mask])
        alpha[mask] = max(alpha[rest], 1 + alpha[rest & ~adj[v] & mask])
    
    # Greedy initialization to get an initial upper bound
    def greedy_upper_bound():
        remaining = full_mask
        parts = []
        while remaining:
            # Find the largest valid subset of remaining
            best_sub = 0
            best_size = 0
            sub = remaining
            while sub:
                if is_valid[sub] and popcount[sub] > best_size:
                    best_sub = sub
                    best_size = popcount[sub]
                sub = (sub - 1) & remaining
            
            if best_sub == 0:
                # Should not happen, but fallback
                lsb = remaining & -remaining
                v = lsb.bit_length() - 1
                parts.append([v])
                remaining ^= lsb
            else:
                part_verts = []
                temp = best_sub
                while temp:
                    b = temp & -temp
                    idx = b.bit_length() - 1
                    part_verts.append(idx)
                    temp ^= b
                parts.append(part_verts)
                remaining ^= best_sub
        
        return parts, len(parts)
    
    greedy_parts, greedy_count = greedy_upper_bound()
    best_parts_count = greedy_count
    best_partition = greedy_parts
    current_parts = []
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 55.0  # 55s margin out of 90s
    
    def search(rem, count):
        nonlocal best_parts_count, best_partition, current_parts
        
        if time.time() > deadline:
            return
        
        if rem == 0:
            if count < best_parts_count:
                best_parts_count = count
                best_partition = [list(part) for part in current_parts]
            return
            
        if count >= best_parts_count:
            return
            
        # Lower bound: at least ceil(popcount(rem) / max(omega[rem], alpha[rem])) parts needed
        max_part_size = max(omega[rem], alpha[rem])
        if max_part_size == 0:
            return
        lower_bound = ceil(popcount[rem] / max_part_size)
        if count + lower_bound >= best_parts_count:
            return
            
        # Choose a vertex to anchor the subset - use the vertex with highest degree in remaining subgraph
        # This helps prune more effectively
        # For simplicity, use LSB but we could improve
        v_bit = rem & -rem
        v = v_bit.bit_length() - 1
        
        # Collect all valid submasks of rem that contain v, sorted by size descending
        candidates = []
        sub = rem
        while sub:
            if (sub & v_bit) and is_valid[sub]:
                candidates.append(sub)
            sub = (sub - 1) & rem
        
        # Sort by popcount descending to find good solutions early
        candidates.sort(key=lambda s: popcount[s], reverse=True)
        
        for sub in candidates:
            if time.time() > deadline:
                return
            
            # Additional pruning: if adding this part doesn't improve lower bound
            new_rem = rem ^ sub
            if new_rem == 0:
                if count + 1 < best_parts_count:
                    best_parts_count = count + 1
                    part_verts = []
                    temp = sub
                    while temp:
                        b = temp & -temp
                        idx = b.bit_length() - 1
                        part_verts.append(idx)
                        temp ^= b
                    current_parts.append(part_verts)
                    best_partition = [list(p) for p in current_parts]
                    current_parts.pop()
                return
            
            new_max_size = max(omega[new_rem], alpha[new_rem])
            if new_max_size == 0:
                continue
            new_lb = ceil(popcount[new_rem] / new_max_size)
            if count + 1 + new_lb >= best_parts_count:
                continue
            
            # Extract vertices for this subset
            part_verts = []
            temp_sub = sub
            while temp_sub:
                b = temp_sub & -temp_sub
                idx = b.bit_length() - 1
                part_verts.append(idx)
                temp_sub ^= b
                
            current_parts.append(part_verts)
            search(new_rem, count + 1)
            current_parts.pop()
            
            # If we found a solution with count+1 parts, we can stop
            # since we're iterating in order of decreasing size, and further
            # smaller subsets won't help with the same count
            if best_parts_count <= count + 1:
                return
    
    search(full_mask, 0)
    
    # Ensure types are correct (int, list)
    if best_partition is None:
        # Fallback
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
    
    return best_partition
