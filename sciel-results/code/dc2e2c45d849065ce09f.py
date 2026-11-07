import time
import random

N = 15
DEADLINE = 85.0  # Safety margin before 90s kill

def _is_valid_fast(adj_mask):
    """
    Check validity using bitmask adjacency.
    adj_mask[u] is an integer bitmask of neighbors of u.
    """
    deg = [0] * N
    for u in range(N):
        deg[u] = bin(adj_mask[u]).count('1')
        
    # Group vertices by degree to speed up checks?
    # Or just iterate edges.
    for u in range(N):
        for v in range(u + 1, N):
            if adj_mask[u] & (1 << v):
                # Edge (u, v) exists
                # Get neighbors of u excluding v
                mask_u = adj_mask[u] & ~(1 << v)
                # Get neighbors of v excluding u
                mask_v = adj_mask[v] & ~(1 << u)
                
                # We need to check if there exist s in mask_u, t in mask_v 
                # such that deg[s] == deg[t] and s != t.
                
                # Optimization: 
                # Let deg_set_u = set of degrees of vertices in mask_u
                # Let deg_set_v = set of degrees of vertices in mask_v
                # If deg_set_u and deg_set_v are disjoint, continue.
                
                # To be precise:
                # For each degree d present in both:
                # Let S_d = {s in mask_u | deg[s] == d}
                # Let T_d = {t in mask_v | deg[t] == d}
                # Violation if we can pick s in S_d, t in T_d with s != t.
                # This fails only if S_d == T_d == {x}.
                
                # Efficient check:
                # Iterate through bits in mask_u?
                # N=15 is small.
                
                # Create lists of (vertex, degree) for mask_u and mask_v
                # Actually, just check pairs?
                # Worst case 14*14 = 196 checks per edge. 105 edges. ~20k ops. Fast enough.
                
                # But we can do better.
                # Precompute degree of each vertex.
                # For this edge, build a map: degree -> bitmask of vertices in mask_u with that degree.
                # And degree -> bitmask of vertices in mask_v with that degree.
                
                deg_map_u = {}
                temp_mask = mask_u
                while temp_mask:
                    # Get lowest set bit
                    lsb = temp_mask & -temp_mask
                    s = lsb.bit_length() - 1
                    d = deg[s]
                    if d not in deg_map_u:
                        deg_map_u[d] = 0
                    deg_map_u[d] |= (1 << s)
                    temp_mask ^= lsb
                    
                deg_map_v = {}
                temp_mask = mask_v
                while temp_mask:
                    lsb = temp_mask & -temp_mask
                    t = lsb.bit_length() - 1
                    d = deg[t]
                    if d not in deg_map_v:
                        deg_map_v[d] = 0
                    deg_map_v[d] |= (1 << t)
                    temp_mask ^= lsb
                    
                # Check intersection
                for d in deg_map_u:
                    if d in deg_map_v:
                        mask_s = deg_map_u[d]
                        mask_t = deg_map_v[d]
                        
                        # Check if we can pick distinct s, t
                        # If mask_s & mask_t is not empty, there is a common vertex x.
                        # If mask_s == (1<<x) and mask_t == (1<<x), then only pair is (x,x), which is OK.
                        # Otherwise, violation.
                        
                        common = mask_s & mask_t
                        if common:
                            # If common is not the entirety of mask_s and mask_t, or if sizes > 1
                            # Violation if NOT (mask_s == common AND mask_t == common)
                            if mask_s != common or mask_t != common:
                                return False
                                
    return True

def _edge_count(adj_mask):
    count = 0
    for u in range(N):
        count += bin(adj_mask[u]).count('1')
    return count // 2

def _try_add(adj_mask, u, v):
    if u == v or (adj_mask[u] & (1 << v)):
        return False
    adj_mask[u] |= (1 << v)
    adj_mask[v] |= (1 << u)
    if _is_valid_fast(adj_mask):
        return True
    adj_mask[u] ^= (1 << v)
    adj_mask[v] ^= (1 << u)
    return False

def _try_swap(adj_mask, u1, v1, u2, v2):
    if not (adj_mask[u1] & (1 << v1)) or (adj_mask[u2] & (1 << v2)):
        return False
    adj_mask[u1] ^= (1 << v1)
    adj_mask[v1] ^= (1 << u1)
    adj_mask[u2] |= (1 << v2)
    adj_mask[v2] |= (1 << u2)
    if _is_valid_fast(adj_mask):
        return True
    adj_mask[u1] ^= (1 << v1)
    adj_mask[v1] ^= (1 << u1)
    adj_mask[u2] ^= (1 << v2)
    adj_mask[v2] ^= (1 << u2)
    return False

def _local_search(adj_mask, start_time, seed):
    random.seed(seed)
    current_count = _edge_count(adj_mask)
    best_mask = adj_mask[:]
    best_count = current_count
    no_improve_rounds = 0
    
    while time.time() - start_time < DEADLINE:
        improved = False
        
        # Try adding missing edges
        missing = [(u, v) for u in range(N) for v in range(u + 1, N) if not (adj_mask[u] & (1 << v))]
        random.shuffle(missing)
        for u, v in missing:
            if time.time() - start_time > DEADLINE:
                break
            if _try_add(adj_mask, u, v):
                current_count += 1
                improved = True
                if current_count > best_count:
                    best_count = current_count
                    best_mask = adj_mask[:]
        
        if improved:
            no_improve_rounds = 0
            continue
        
        no_improve_rounds += 1
        if no_improve_rounds > 10:
            break
        
        # Try swaps
        existing = [(u, v) for u in range(N) for v in range(u + 1, N) if (adj_mask[u] & (1 << v))]
        missing = [(u, v) for u in range(N) for v in range(u + 1, N) if not (adj_mask[u] & (1 << v))]
        
        if not missing:
            break
            
        random.shuffle(existing)
        random.shuffle(missing)
        
        found_swap = False
        # Limit swaps tried to avoid slow rounds
        max_swaps_to_try = 1000
        tried = 0
        for u1, v1 in existing:
            if time.time() - start_time > DEADLINE or tried > max_swaps_to_try:
                break
            for u2, v2 in missing:
                if _try_swap(adj_mask, u1, v1, u2, v2):
                    found_swap = True
                    break
                tried += 1
            if found_swap or tried > max_swaps_to_try:
                break
        
        if not found_swap:
            # If no swap found in limited tries, might be stuck. 
            # But we break out of main loop if no_improve_rounds > 10.
            # To be safe, if we didn't find a swap in this round, we increment no_improve?
            # Actually, if no swap found, we didn't improve.
            pass 

    return best_mask, best_count

def _bipartite_start(a_size):
    adj_mask = [0] * N
    for u in range(a_size):
        for v in range(a_size, N):
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
    return adj_mask

def construct():
    start_time = time.time()
    best_mask = [0] * N
    best_count = 0
    
    # Try different bipartite splits
    for a_size in [7, 8, 6, 9, 5, 10]:
        if time.time() - start_time > DEADLINE - 5:
            break
        adj_mask = _bipartite_start(a_size)
        count = _edge_count(adj_mask)
        
        adj_ls, count_ls = _local_search(adj_mask, start_time, seed=a_size * 100)
        if count_ls > best_count:
            best_count = count_ls
            best_mask = adj_ls
    
    # Try random greedy starts
    for i in range(10):
        if time.time() - start_time > DEADLINE - 5:
            break
        random.seed(10000 + i)
        adj_mask = [0] * N
        edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
        random.shuffle(edges)
        for u, v in edges:
            if time.time() - start_time > DEADLINE - 2:
                break
            _try_add(adj_mask, u, v)
        count = _edge_count(adj_mask)
        if count > best_count:
            adj_ls, count_ls = _local_search(adj_mask, start_time, seed=20000 + i)
            if count_ls > best_count:
                best_count = count_ls
                best_mask = adj_ls
    
    # Final validation
    if not _is_valid_fast(best_mask):
        best_mask = _bipartite_start(7)
        best_count = _edge_count(best_mask)
    
    result = []
    for u in range(N):
        for v in range(u + 1, N):
            if best_mask[u] & (1 << v):
                result.append([int(u), int(v)])
    
    return result
