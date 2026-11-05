import time
import random

def construct():
    """
    Build a C4-free graph on n vertices with min degree >= n - 39.
    Return {'n': n, 'edges': [[u, v], ...]}.
    Maximize n.
    """
    deadline = time.time() + 140  # Safety margin
    
    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        if time.time() > deadline:
            break
        
        required_min_deg = n - 39
        
        # Try multiple random restarts
        for attempt in range(1000):
            if time.time() > deadline:
                break
            
            result = _construct_c4_free(n, required_min_deg, deadline, attempt)
            if result is not None:
                return result
    
    # Fallback: return a valid small graph
    # For n=39, min degree >= 0, so empty graph works
    return {'n': 39, 'edges': []}

def _check_c4_bitset(adj, u, v):
    """
    Check if adding edge (u, v) creates a C4.
    adj: list of integers representing bitsets of neighbors.
    Returns True if C4 would be created.
    """
    # Calculate set of vertices at distance 2 from u
    S_u = 0
    mask_u = 1 << u
    mask_v = 1 << v
    
    # Iterate neighbors of u
    neighbors_u = adj[u]
    while neighbors_u:
        lsb = neighbors_u & -neighbors_u
        x = lsb.bit_length() - 1
        neighbors_u ^= lsb
        S_u |= adj[x]
        
    # Check if there is a y in N(v) that is in S_u but not in N(u)
    # If y is in N(u), it's a triangle, which is allowed.
    # If y is not in N(u), then there is x in N(u) with x~y and x!=y, forming C4 u-x-y-v-u.
    candidates = S_u & adj[v] & ~mask_u & ~mask_v
    if candidates:
        return True
        
    return False

def _construct_c4_free(n, min_deg_req, deadline, seed):
    rng = random.Random(seed)
    
    # Initialize empty graph
    adj = [0] * n
    
    # Phase 1: Greedy construction
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    rng.shuffle(all_edges)
    
    for u, v in all_edges:
        if time.time() > deadline:
            break
        if not _check_c4_bitset(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
    # Phase 2: Local Search
    def get_degrees(adj):
        return [bin(x).count('1') for x in adj]
    
    degrees = get_degrees(adj)
    
    max_iters = 2000
    for _ in range(max_iters):
        if time.time() > deadline:
            break
            
        degrees = get_degrees(adj)
        min_deg = min(degrees)
        
        if min_deg >= min_deg_req:
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u] & (1 << v):
                        edges.append([u, v])
            return {'n': n, 'edges': edges}
        
        # Find vertices with minimum degree
        min_verts = [i for i, d in enumerate(degrees) if d == min_deg]
        v = rng.choice(min_verts)
        
        # Try to add an edge to v
        added = False
        non_neighbors = [u for u in range(n) if u != v and not (adj[v] & (1 << u))]
        rng.shuffle(non_neighbors)
        
        for u in non_neighbors:
            if not _check_c4_bitset(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                added = True
                break
        
        if added:
            continue
            
        # If no direct addition, try a swap
        current_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    current_edges.append((u, v))
                    
        if not current_edges:
            break
            
        a, b = rng.choice(current_edges)
        
        # Save state
        adj[a] ^= (1 << b)
        adj[b] ^= (1 << a)
        
        # Try to add edges to v (and other low degree vertices)
        old_degrees = get_degrees(adj)
        old_min = min(old_degrees)
        
        non_neighbors_v = [u for u in range(n) if u != v and not (adj[v] & (1 << u))]
        rng.shuffle(non_neighbors_v)
        
        added_any = False
        added_u = -1
        for u in non_neighbors_v:
            if not _check_c4_bitset(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                added_any = True
                added_u = u
                break
                
        if added_any:
            new_degrees = get_degrees(adj)
            new_min = min(new_degrees)
            
            if new_min > old_min:
                # Keep swap
                pass
            else:
                # Revert
                if added_u != -1:
                    adj[added_u] ^= (1 << v)
                    adj[v] ^= (1 << added_u)
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
        else:
            # Revert removal
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)

    # Final check
    degrees = get_degrees(adj)
    if min(degrees) >= min_deg_req:
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    edges.append([u, v])
        return {'n': n, 'edges': edges}
    
    return None
