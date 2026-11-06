<<<<<<< SEARCH
def _max_independent_set_branch_and_bound(adj, deadline, rng, min_edges=0):
    """
    Branch and bound to find maximum edge count triangle-free graph
    that is NOT 3-colorable, starting from adj.
    """
    best_cnt = min_edges
    best_adj = None
    
    # Get all possible edges
    all_edges = []
    for u in range(N):
        for v in range(u + 1, N):
            all_edges.append((u, v))
    
    # Sort edges by some heuristic (e.g., sum of degrees)
    def edge_score(e):
        u, v = e
        return bin(adj[u]).count('1') + bin(adj[v]).count('1')
    
    # We'll use a randomized local search with occasional restarts
    # combined with branch-and-bound on promising candidates
    
    current_adj = [a for a in adj]
    current_cnt = _edge_count(current_adj)
    
    if current_cnt > best_cnt and not _is_3_colorable(current_adj):
        best_cnt = current_cnt
        best_adj = [a for a in current_adj]
    
    # Local search phase
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try to add edges
        missing = [(u, v) for u in range(N) for v in range(u + 1, N) 
                   if not ((current_adj[u] >> v) & 1)]
        rng.shuffle(missing)
        
        for u, v in missing:
            if time.time() > deadline:
                break
            if _is_triangle_free_add(current_adj, u, v):
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                current_cnt += 1
                improved = True
                
                if current_cnt > best_cnt and not _is_3_colorable(current_adj):
                    best_cnt = current_cnt
                    best_adj = [a for a in current_adj]
                break
        
        if not improved:
            # Try edge swaps: remove one edge, add two
            existing = [(u, v) for u in range(N) for v in range(u + 1, N) 
                       if (current_adj[u] >> v) & 1]
            if not existing:
                break
            
            best_swap = None
            best_swap_cnt = current_cnt
            
            for _ in range(10):
                if time.time() > deadline:
                    break
                rem_u, rem_v = rng.choice(existing)
                
                adj_copy = [a for a in current_adj]
                adj_copy[rem_u] &= ~(1 << rem_v)
                adj_copy[rem_v] &= ~(1 << rem_u)
                
                # Try to add edges greedily
                for u in range(N):
                    for v in range(u + 1, N):
                        if not ((adj_copy[u] >> v) & 1) and _is_triangle_free_add(adj_copy, u, v):
                            adj_copy[u] |= (1 << v)
                            adj_copy[v] |= (1 << u)
                
                cnt = _edge_count(adj_copy)
                if cnt > best_swap_cnt and not _is_3_colorable(adj_copy):
                    best_swap_cnt = cnt
                    best_swap = adj_copy
            
            if best_swap:
                current_adj = best_swap
                current_cnt = best_swap_cnt
                improved = True
                if current_cnt > best_cnt:
                    best_cnt = current_cnt
                    best_adj = [a for a in current_adj]
    
    return best_adj, best_cnt
=======
def _greedy_add_all(adj):
    """Greedily add all possible triangle-free edges."""
    changed = True
    while changed:
        changed = False
        for u in range(N):
            for v in range(u + 1, N):
                if not ((adj[u] >> v) & 1) and _is_triangle_free_add(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    changed = True
    return adj

def _try_swap_remove_add2(adj, rem_u, rem_v, deadline):
    """Remove edge (rem_u, rem_v), then greedily add up to 2 edges."""
    adj_copy = [a for a in adj]
    adj_copy[rem_u] &= ~(1 << rem_v)
    adj_copy[rem_v] &= ~(1 << rem_u)
    
    # Find all edges that can now be added
    addable = []
    for u in range(N):
        for v in range(u + 1, N):
            if not ((adj_copy[u] >> v) & 1) and _is_triangle_free_add(adj_copy, u, v):
                addable.append((u, v))
    
    if not addable:
        return None, 0
    
    # Try adding 2 edges (or 1 if only 1 available)
    best_result = None
    best_cnt = -1
    
    if len(addable) >= 2:
        # Try all pairs of addable edges (limited for speed)
        rng_local = random.Random(hash((rem_u, rem_v)) % 10000)
        rng_local.shuffle(addable)
        max_pairs = min(len(addable) * (len(addable) - 1) // 2, 50)
        count = 0
        for i in range(min(len(addable), 20)):
            for j in range(i + 1, min(len(addable), 20)):
                if count >= max_pairs or time.time() > deadline:
                    break
                count += 1
                e1, e2 = addable[i], addable[j]
                # Check no conflict between the two new edges
                u1, v1 = e1
                u2, v2 = e2
                # They can't share a vertex in a way that creates issues
                # Just try adding both
                test_adj = [a for a in adj_copy]
                test_adj[u1] |= (1 << v1)
                test_adj[v1] |= (1 << u1)
                if _is_triangle_free_add(test_adj, u2, v2):
                    test_adj[u2] |= (1 << v2)
                    test_adj[v2] |= (1 << u2)
                    cnt = _edge_count(test_adj)
                    if cnt > best_cnt and not _is_3_colorable(test_adj):
                        best_cnt = cnt
                        best_result = test_adj
            if time.time() > deadline:
                break
    else:
        # Only 1 edge can be added
        u, v = addable[0]
        adj_copy[u] |= (1 << v)
        adj_copy[v] |= (1 << u)
        cnt = _edge_count(adj_copy)
        if cnt > best_cnt and not _is_3_colorable(adj_copy):
            best_cnt = cnt
            best_result = adj_copy
    
    return best_result, best_cnt

def _local_search(adj, deadline, rng, min_edges=0):
    """
    Aggressive local search to find maximum edge count triangle-free graph
    that is NOT 3-colorable, starting from adj.
    """
    best_cnt = min_edges
    best_adj = None
    
    current_adj = [a for a in adj]
    _greedy_add_all(current_adj)
    current_cnt = _edge_count(current_adj)
    
    if current_cnt > best_cnt and not _is_3_colorable(current_adj):
        best_cnt = current_cnt
        best_adj = [a for a in current_adj]
    
    # Local search phase
    stall_count = 0
    max_stall = 30
    
    while time.time() < deadline:
        improved = False
        stall_count += 1
        
        if stall_count > max_stall:
            # Restart with perturbation
            current_adj = [a for a in best_adj] if best_adj else [a for a in adj]
            # Randomly remove 1-3 edges
            existing = [(u, v) for u in range(N) for v in range(u + 1, N) 
                       if (current_adj[u] >> v) & 1]
            if existing:
                for _ in range(rng.randint(1, 3)):
                    if not existing:
                        break
                    idx = rng.randint(0, len(existing) - 1)
                    u, v = existing[idx]
                    current_adj[u] &= ~(1 << v)
                    current_adj[v] &= ~(1 << u)
                    existing = [(uu, vv) for uu in range(N) for vv in range(uu + 1, N) 
                               if (current_adj[uu] >> vv) & 1]
            _greedy_add_all(current_adj)
            current_cnt = _edge_count(current_adj)
            stall_count = 0
            continue
        
        # Phase 1: Try to add single edges
        missing = [(u, v) for u in range(N) for v in range(u + 1, N) 
                   if not ((current_adj[u] >> v) & 1)]
        rng.shuffle(missing)
        
        for u, v in missing:
            if time.time() > deadline:
                break
            if _is_triangle_free_add(current_adj, u, v):
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                current_cnt += 1
                improved = True
                
                if current_cnt > best_cnt and not _is_3_colorable(current_adj):
                    best_cnt = current_cnt
                    best_adj = [a for a in current_adj]
                break
        
        if improved:
            stall_count = 0
            continue
        
        # Phase 2: Try remove-1-add-2 swaps
        existing = [(u, v) for u in range(N) for v in range(u + 1, N) 
                   if (current_adj[u] >> v) & 1]
        
        if existing:
            rng.shuffle(existing)
            found_swap = False
            
            for rem_u, rem_v in existing:
                if time.time() > deadline:
                    break
                if found_swap:
                    break
                
                result_adj, result_cnt = _try_swap_remove_add2(current_adj, rem_u, rem_v, deadline)
                
                if result_adj and result_cnt > current_cnt:
                    current_adj = result_adj
                    current_cnt = result_cnt
                    improved = True
                    found_swap = True
                    
                    if current_cnt > best_cnt and not _is_3_colorable(current_adj):
                        best_cnt = current_cnt
                        best_adj = [a for a in current_adj]
            
            if improved:
                stall_count = 0
                continue
        
        # Phase 3: Try remove-2-add-3 swaps (more aggressive)
        if len(existing) >= 2:
            rng.shuffle(existing)
            found_swap2 = False
            
            for i in range(min(len(existing), 15)):
                if time.time() > deadline or found_swap2:
                    break
                rem1_u, rem1_v = existing[i]
                for j in range(i + 1, min(len(existing), 15)):
                    if time.time() > deadline or found_swap2:
                        break
                    rem2_u, rem2_v = existing[j]
                    
                    adj_copy = [a for a in current_adj]
                    adj_copy[rem1_u] &= ~(1 << rem1_v)
                    adj_copy[rem1_v] &= ~(1 << rem1_u)
                    adj_copy[rem2_u] &= ~(1 << rem2_v)
                    adj_copy[rem2_v] &= ~(1 << rem2_u)
                    
                    _greedy_add_all(adj_copy)
                    cnt = _edge_count(adj_copy)
                    
                    if cnt > current_cnt and not _is_3_colorable(adj_copy):
                        current_adj = adj_copy
                        current_cnt = cnt
                        improved = True
                        found_swap2 = True
                        
                        if current_cnt > best_cnt:
                            best_cnt = current_cnt
                            best_adj = [a for a in current_adj]
                        break
            
            if improved:
                stall_count = 0
                continue
        
        if not improved:
            stall_count += 1
    
    return best_adj, best_cnt
>>>>>>> REPLACE

