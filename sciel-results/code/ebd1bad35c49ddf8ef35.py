import numpy as np
import time
import random

# EVOLVE-BLOCK-START
_N_CACHE = {}

def _fast_c4_check(adj, u, v):
    """
    Fast C4 check using bitsets (int masks).
    Returns True if adding (u, v) creates a C4.
    """
    # For each common neighbor? No, C4 is u-x-y-v.
    # Need to check if there exists an edge (x, y) where x in N(u)\{v}, y in N(v)\{u}.
    # We can check if any x in N(u)\{v} has a neighbor in N(v)\{u}.
    
    mask_u = adj[u] & ~(1 << v)
    mask_v = adj[v] & ~(1 << u)
    
    # Iterate through neighbors of u (excluding v)
    # For each such neighbor x, check if x has any neighbor in mask_v
    # To speed up, we can check if (adj[x] & mask_v) != 0 for any x in mask_u
    
    # Since n <= 47, we can iterate through set bits in mask_u
    # But checking all x in mask_u might be slow if mask_u is large.
    # Alternative: Check intersection of neighborhoods?
    
    # Let's stick to iterating set bits in mask_u.
    # We can optimize by checking the smaller set first?
    # If popcount(mask_u) > popcount(mask_v), iterate mask_v instead?
    # Symmetric: exists x in mask_u, y in mask_v s.t. adj[x,y]=True.
    
    # Let's just iterate mask_u.
    m = mask_u
    while m:
        # Get lowest set bit
        lsb = m & (-m)
        x = lsb.bit_length() - 1
        
        # Check if x is connected to any node in mask_v
        if adj[x] & mask_v:
            return True
        
        m ^= lsb
        
    return False

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
        max_attempts = 5000
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Use a mix of seeds based on attempt to ensure variety
            # The original used attempt as seed. We can keep that or vary it.
            result = _try_construct_fast(n, required_min_deg, deadline, attempt)
            if result is not None:
                n_found, edges_found = result
                return {'n': n_found, 'edges': edges_found}
        
    # Fallback: return a valid smaller graph
    # For n=39, min degree >= 0, so empty graph works
    return {'n': 39, 'edges': []}

# Keep old function for compatibility if called, but we will use _fast_c4_check in the new _try_construct_fast
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
    Greedy construction with efficient C4 checking using bitsets.
    """
    rng = np.random.RandomState(seed)
    
    # Use integer bitmasks for adjacency
    # adj[i] is an integer where bit j is 1 if (i,j) is an edge
    adj = [0] * n
    
    # Track degrees
    degrees = [0] * n
    
    # Helper to get mask of neighbors of u excluding v
    def get_mask_excl(u, v):
        return adj[u] & ~(1 << v)
    
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
        if min(degrees) >= required_min_deg:
            break
            
        # Find vertices that need more edges
        needy = [i for i in range(n) if degrees[i] < required_min_deg]
        if len(needy) == 0:
            break
            
        # Pick a random needy vertex
        u = needy[rng.randint(0, len(needy))]
        
        # Find candidates for u: non-neighbors
        # adj[u] has bits set for neighbors. We want bits NOT set.
        # All vertices mask
        all_mask = (1 << n) - 1
        # Non-neighbors of u
        non_neighbors_mask = all_mask & ~adj[u]
        # Exclude self
        non_neighbors_mask &= ~(1 << u)
        
        if non_neighbors_mask == 0:
            # u is connected to everyone, but degree < required? Impossible if n is small.
            # Or we are stuck.
            return None
            
        # Get list of candidate vertices
        candidates = []
        m = non_neighbors_mask
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            candidates.append(v)
            m ^= lsb
            
        if not candidates:
            return None
            
        # Sort candidates by degree (ascending) to balance the graph
        # This helps in reaching min degree faster
        candidates.sort(key=lambda x: degrees[x])
        
        added = False
        # Try candidates in sorted order, but with some randomness at the top
        # to avoid deterministic failure patterns
        # Let's just try them in sorted order first. If that fails, we might need backtracking or restart.
        # For now, deterministic sorted order within the random restart framework.
        
        for v in candidates:
            if _fast_c4_check(adj, u, v):
                continue
                
            # Safe to add
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
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
                non_neighbors_mask2 = all_mask & ~adj[u2]
                non_neighbors_mask2 &= ~(1 << u2)
                
                if non_neighbors_mask2 == 0:
                    continue
                
                candidates2 = []
                m2 = non_neighbors_mask2
                while m2:
                    lsb2 = m2 & (-m2)
                    v2 = lsb2.bit_length() - 1
                    candidates2.append(v2)
                    m2 ^= lsb2
                    
                if not candidates2:
                    continue
                
                candidates2.sort(key=lambda x: degrees[x])
                
                for v2 in candidates2:
                    if not _fast_c4_check(adj, u2, v2):
                        adj[u2] |= (1 << v2)
                        adj[v2] |= (1 << u2)
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
            if adj[u] & (1 << v):
                edges.append([u, v])
                
    return (n, edges)

# EVOLVE-BLOCK-END
