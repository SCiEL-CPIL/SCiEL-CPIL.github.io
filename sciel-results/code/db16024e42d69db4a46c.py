import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Seed with Paley 17
    best_n = 17
    best_edges = _paley_17()
    
    # Try incremental construction from larger to smaller
    for n in range(35, 17, -1):
        if time.time() > deadline:
            break
        time_alloc = min(20.0, (deadline - time.time()) * 0.8)
        if time_alloc < 2.0:
            break
        success, edges = _incremental_construct(n, time_alloc, deadline)
        if success:
            best_n = n
            best_edges = edges
            if n >= 35:
                break
                
    return {'n': best_n, 'edges': best_edges}

def _paley_17():
    squares = {x * x % 17 for x in range(1, 17)}
    return [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]

def _incremental_construct(n, time_alloc, deadline):
    """
    Build graph vertex by vertex. At each step, add a new vertex and choose
    its connections to previous vertices such that:
    1. No K4 is created
    2. No independent set of size 6 is created
    
    We represent the graph as a list of bitmasks (one per vertex).
    """
    best_result = None
    best_n_found = 0
    
    num_restarts = max(1, int(time_alloc / 5.0))
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        adj_masks = [0] * n  # adj_masks[i] is a bitmask of neighbors of i
        
        # Start with a small seed graph (e.g., 5 vertices)
        seed_n = min(5, n)
        for i in range(seed_n):
            for j in range(i + 1, seed_n):
                if random.random() < 0.5:
                    adj_masks[i] |= (1 << j)
                    adj_masks[j] |= (1 << i)
        
        # Verify seed is valid
        if not _is_valid_masks(adj_masks, seed_n):
            continue
        
        # Incrementally add vertices
        for v in range(seed_n, n):
            if time.time() > deadline - 2:
                break
            
            # Try multiple random connection patterns
            found_valid = False
            for attempt in range(200):
                # Choose random connections to previous vertices
                new_mask = 0
                for u in range(v):
                    if random.random() < 0.5:
                        new_mask |= (1 << u)
                
                # Check if adding vertex v with new_mask maintains validity
                if _check_add_vertex(adj_masks, v, new_mask):
                    # Add the vertex
                    adj_masks[v] = new_mask
                    for u in range(v):
                        if new_mask & (1 << u):
                            adj_masks[u] |= (1 << v)
                    found_valid = True
                    break
            
            if not found_valid:
                # Try with different random seeds or break
                break
        
        # Check if we got a valid graph on n vertices
        if len(adj_masks) == n and _is_valid_masks(adj_masks, n):
            edges = _masks_to_edges(adj_masks, n)
            return True, edges
    
    return False, None

def _is_valid_masks(adj_masks, n):
    """Check if the graph represented by adj_masks has no K4 and no independent 6-set."""
    # Check K4: for any 4 vertices, not all pairwise connected
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj_masks[i] & (1 << j)):
                continue
            for k in range(j + 1, n):
                if not (adj_masks[i] & (1 << k)) or not (adj_masks[j] & (1 << k)):
                    continue
                for l in range(k + 1, n):
                    if (adj_masks[i] & (1 << l)) and (adj_masks[j] & (1 << l)) and (adj_masks[k] & (1 << l)):
                        return False
    # Check independent 6-set: for any 6 vertices, at least one pair is connected
    # Equivalent: complement has no K6
    for i in range(n):
        for j in range(i + 1, n):
            if adj_masks[i] & (1 << j):
                continue
            for k in range(j + 1, n):
                if (adj_masks[i] & (1 << k)) or (adj_masks[j] & (1 << k)):
                    continue
                for l in range(k + 1, n):
                    if (adj_masks[i] & (1 << l)) or (adj_masks[j] & (1 << l)) or (adj_masks[k] & (1 << l)):
                        continue
                    for m in range(l + 1, n):
                        if (adj_masks[i] & (1 << m)) or (adj_masks[j] & (1 << m)) or (adj_masks[k] & (1 << m)) or (adj_masks[l] & (1 << m)):
                            continue
                        for p in range(m + 1, n):
                            if (adj_masks[i] & (1 << p)) or (adj_masks[j] & (1 << p)) or (adj_masks[k] & (1 << p)) or (adj_masks[l] & (1 << p)) or (adj_masks[m] & (1 << p)):
                                return False
    return True

def _check_add_vertex(adj_masks, v, new_mask):
    """Check if adding vertex v with connections new_mask maintains K4-free and no-Ind6."""
    # Check K4: for any triangle among previous vertices that v connects to, 
    # all three must not be mutually connected (otherwise v+triangle = K4)
    # More precisely: for any i<j<k all connected to v, if i-j, i-k, j-k are all edges, that's a K4
    for i in range(v):
        if not (new_mask & (1 << i)):
            continue
        for j in range(i + 1, v):
            if not (new_mask & (1 << j)):
                continue
            for k in range(j + 1, v):
                if not (new_mask & (1 << k)):
                    continue
                # Check if i,j,k form a triangle
                if (adj_masks[i] & (1 << j)) and (adj_masks[i] & (1 << k)) and (adj_masks[j] & (1 << k)):
                    return False
    
    # Check Ind6: for any 5 vertices among previous that are NOT connected to v,
    # they must not be mutually independent (otherwise v+those 5 = Ind6)
    # More precisely: find all u where new_mask & (1<<u) == 0
    non_neighbors = []
    for u in range(v):
        if not (new_mask & (1 << u)):
            non_neighbors.append(u)
    
    # Check if non_neighbors contains an independent set of size 5
    if len(non_neighbors) >= 5:
        if _has_independent_5_in_subset(adj_masks, non_neighbors):
            return False
    
    return True

def _has_independent_5_in_subset(adj_masks, subset):
    """Check if the induced subgraph on subset has an independent set of size 5."""
    if len(subset) < 5:
        return False
    # Brute force for small subsets
    if len(subset) <= 15:
        # Check all 5-combinations
        from itertools import combinations
        for combo in combinations(subset, 5):
            # Check if all pairs in combo are non-adjacent
            is_independent = True
            for i in range(5):
                for j in range(i + 1, 5):
                    if adj_masks[combo[i]] & (1 << combo[j]):
                        is_independent = False
                        break
                if not is_independent:
                    break
            if is_independent:
                return True
        return False
    else:
        # For larger subsets, use a greedy heuristic or return False conservatively
        # This is a heuristic - may miss some cases but is fast
        # Try to find 5 independent vertices greedily
        chosen = []
        for v in subset:
            can_add = True
            for u in chosen:
                if adj_masks[v] & (1 << u):
                    can_add = False
                    break
            if can_add:
                chosen.append(v)
            if len(chosen) >= 5:
                return True
        return False

def _masks_to_edges(adj_masks, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj_masks[i] & (1 << j):
                edges.append([i, j])
    return edges
