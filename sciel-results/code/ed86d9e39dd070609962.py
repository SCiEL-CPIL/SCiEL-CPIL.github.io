import numpy as np
import time
import random

# EVOLVE-BLOCK-START

def construct():
    """
    Try to build a C4-free graph on n vertices (n up to 47) with min degree >= n - 39.
    Return {'n': n, 'edges': [[u, v], ...]}.
    Maximize n.
    """
    deadline = time.time() + 145  # Leave 5s margin for 150s limit
    
    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        if time.time() > deadline:
            break
        
        required_min_deg = n - 39
        
        # Try multiple random restarts
        # With the fast C4 check, we can do many more attempts
        max_attempts = 2000
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            result = _try_construct_fast(n, required_min_deg, deadline, attempt)
            if result is not None:
                n_found, edges_found = result
                return {'n': n_found, 'edges': edges_found}
        
        # If greedy fails, try a simple local search on the best attempt?
        # For now, rely on restarts.
        
    # Fallback: return a valid smaller graph
    # For n=39, min degree >= 0, so empty graph works
    return {'n': 39, 'edges': []}

def _check_c4_add(adj, u, v):
    """
    Check if adding edge (u,v) creates a C4 in the graph defined by adj.
    A C4 is u-x-y-v-u.
    This exists if there is an edge (x,y) such that x is in N(u)\{v} and y is in N(v)\{u}.
    """
    # N(u) is adj[u], N(v) is adj[v]
    # We need to check if there is any edge between N(u)\{v} and N(v)\{u}
    
    # Create masks
    # mask_u: True for neighbors of u, except v
    mask_u = adj[u].copy()
    mask_u[v] = False
    
    # mask_v: True for neighbors of v, except u
    mask_v = adj[v].copy()
    mask_v[u] = False
    
    # If either set is empty, no C4 can be formed via this specific structure?
    # Wait, if N(u) is empty, we can't form u-x-y-v.
    # So if not np.any(mask_u) or not np.any(mask_v): return False
    
    # Check if any element in mask_u has an edge to any element in mask_v
    # We can extract the submatrix
    # idx_u = np.where(mask_u)[0]
    # idx_v = np.where(mask_v)[0]
    # if len(idx_u) == 0 or len(idx_v) == 0: return False
    # return np.any(adj[np.ix_(idx_u, idx_v)])
    
    # Vectorized approach without index extraction (faster for small n)
    # We want to know if sum_{x,y} adj[x,y] * mask_u[x] * mask_v[y] > 0
    # This is equivalent to checking if the intersection of the neighbor sets of any x in N(u)
    # with N(v) is non-empty? No.
    
    # Let's stick to the index method, it's clean and fast enough for n=47
    idx_u = np.nonzero(mask_u)[0]
    if len(idx_u) == 0:
        return False
    idx_v = np.nonzero(mask_v)[0]
    if len(idx_v) == 0:
        return False
        
    # Check submatrix
    return np.any(adj[np.ix_(idx_u, idx_v)])

def _try_construct_fast(n, required_min_deg, deadline, seed):
    """
    Greedy construction with efficient C4 checking.
    """
    rng = np.random.RandomState(seed)
    
    adj = np.zeros((n, n), dtype=bool)
    
    # Track degrees
    degrees = np.zeros(n, dtype=int)
    
    # List of available edges? No, just iterate.
    # To add randomness, we can shuffle the order of vertex consideration or edge consideration.
    
    # We will iterate until no more edges can be added or min degree is satisfied
    # To avoid getting stuck in a bad local optimum, we use a "randomized greedy" approach:
    # 1. Identify vertices with degree < required_min_deg.
    # 2. Pick one randomly.
    # 3. Try to add an edge to a non-neighbor that doesn't create C4.
    #    Prioritize neighbors that are also low-degree.
    
    # Pre-compute all possible edges for random sampling?
    # Or just loop.
    
    while True:
        if time.time() > deadline:
            return None
            
        # Check if we are done
        if degrees.min() >= required_min_deg:
            break
            
        # Find vertices that need more edges
        needy = np.where(degrees < required_min_deg)[0]
        if len(needy) == 0:
            break
            
        # Pick a random needy vertex
        u = needy[rng.randint(0, len(needy))]
        
        # Find candidates for u: non-neighbors
        # adj[u] is True for neighbors. We want False.
        candidates = np.where(~adj[u])[0]
        candidates = candidates[candidates != u] # Exclude self
        
        if len(candidates) == 0:
            # u is connected to everyone, but degree < required? Impossible if n is small.
            # Or we are stuck.
            return None
            
        # Sort candidates by degree (ascending) to balance the graph
        # This helps in reaching min degree faster
        cand_degrees = degrees[candidates]
        sorted_indices = np.argsort(cand_degrees)
        
        added = False
        # Try candidates in sorted order, but with some randomness at the top
        # to avoid deterministic failure patterns
        # Let's just try them in sorted order first. If that fails, we might need backtracking or restart.
        # For now, deterministic sorted order within the random restart framework.
        
        for idx in sorted_indices:
            v = candidates[idx]
            
            if _check_c4_add(adj, u, v):
                continue
                
            # Safe to add
            adj[u, v] = True
            adj[v, u] = True
            degrees[u] += 1
            degrees[v] += 1
            added = True
            break
            
        if not added:
            # Could not add any edge to u without creating C4
            # This means u is "maximal" with respect to C4-free property?
            # If we can't add any edge to ANY needy vertex, we are stuck.
            # Let's check if we can add to any other needy vertex.
            # Actually, if we picked u randomly and failed, maybe another needy vertex works.
            # But if the graph is saturated (no edges can be added anywhere without C4), we are stuck.
            
            # Check if any edge can be added anywhere
            # This is expensive. Let's just assume if we fail for u, we might fail for others.
            # To be safe, let's try a few other needy vertices.
            
            success = False
            for _ in range(10):
                u2 = needy[rng.randint(0, len(needy))]
                candidates2 = np.where(~adj[u2])[0]
                candidates2 = candidates2[candidates2 != u2]
                if len(candidates2) == 0:
                    continue
                cand_degrees2 = degrees[candidates2]
                sorted_indices2 = np.argsort(cand_degrees2)
                
                for idx2 in sorted_indices2:
                    v2 = candidates2[idx2]
                    if not _check_c4_add(adj, u2, v2):
                        adj[u2, v2] = True
                        adj[v2, u2] = True
                        degrees[u2] += 1
                        degrees[v2] += 1
                        success = True
                        break
                if success:
                    break
                    
            if not success:
                # Stuck
                return None

    # Extract edges
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v]:
                edges.append([u, v])
                
    return (n, edges)

# EVOLVE-BLOCK-END
