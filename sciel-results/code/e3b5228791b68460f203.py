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
        if required_min_deg < 0:
            required_min_deg = 0
        
        if required_min_deg == 0:
            return {'n': n, 'edges': []}
        
        # Try multiple random seeds with polarity graph initialization
        for seed in range(200):
            if time.time() > deadline:
                break
            
            result = _construct_c4_free(n, required_min_deg, deadline, seed)
            if result is not None:
                return result
    
    # Fallback: return a valid small graph
    return {'n': 39, 'edges': []}


def _polarity_seed(n, rng):
    """
    Create a polarity graph-like seed: a C4-free graph with good degree distribution.
    Uses a greedy construction biased to avoid C4 while building connectivity.
    """
    adj = [0] * n
    degrees = [0] * n
    
    # Start with a random matching to give everyone some degree
    perm = list(range(n))
    rng.shuffle(perm)
    for i in range(0, n - 1, 2):
        u, v = perm[i], perm[i + 1]
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        degrees[u] += 1
        degrees[v] += 1
    
    # Add more edges greedily, prioritizing low-degree vertices
    # Use a priority queue-like approach: repeatedly pick the lowest-degree vertex
    # and try to connect it to a non-neighbor without creating C4
    
    for _ in range(n * 3):  # Try to add up to 3n edges
        if time.time() > deadline:
            break
        
        # Find lowest degree vertex
        min_deg = min(degrees)
        low_verts = [i for i in range(n) if degrees[i] == min_deg]
        v = rng.choice(low_verts)
        
        # Try to add an edge from v to a non-neighbor
        non_neighbors = [u for u in range(n) if u != v and not (adj[v] & (1 << u))]
        rng.shuffle(non_neighbors)
        
        added = False
        for u in non_neighbors:
            if _check_c4_bitset(adj, u, v):
                continue
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            degrees[u] += 1
            degrees[v] += 1
            added = True
            break
        
        if not added:
            break
    
    return adj, degrees


def _check_c4_bitset(adj, u, v):
    """
    Check if adding edge (u, v) creates a C4.
    adj: list of integers representing bitsets of neighbors.
    Returns True if C4 would be created.
    """
    mask_u = 1 << u
    mask_v = 1 << v
    
    # For each neighbor x of u (excluding v), check if N(x) intersects N(v) excluding u and v
    neighbors_u = adj[u] & ~mask_v
    
    while neighbors_u:
        lsb = neighbors_u & -neighbors_u
        x = lsb.bit_length() - 1
        neighbors_u ^= lsb
        
        # Check if N(x) ∩ N(v) contains any vertex other than u and v
        common = adj[x] & adj[v]
        common &= ~mask_u
        common &= ~mask_v
        
        if common:
            return True
            
    return False


def _construct_c4_free(n, min_deg_req, deadline, seed):
    rng = random.Random(seed)
    
    # Phase 1: Create a good initial seed using polarity-like construction
    adj, degrees = _polarity_seed(n, rng)
    
    # Phase 2: Targeted local search to boost minimum degree
    # Repeatedly identify the lowest-degree vertex and try to add edges to it
    
    max_iters = 5000
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Check if we've met the requirement
        min_deg = min(degrees)
        if min_deg >= min_deg_req:
            # Success - extract edges
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u] & (1 << v):
                        edges.append([u, v])
            return {'n': n, 'edges': edges}
        
        # Find vertices with minimum degree
        min_deg = min(degrees)
        min_verts = [i for i, d in enumerate(degrees) if d == min_deg]
        v = rng.choice(min_verts)
        
        # Try to add an edge to v from a non-neighbor
        non_neighbors = [u for u in range(n) if u != v and not (adj[v] & (1 << u))]
        rng.shuffle(non_neighbors)
        
        added = False
        for u in non_neighbors:
            if not _check_c4_bitset(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                degrees[u] += 1
                degrees[v] += 1
                added = True
                break
        
        if not added:
            # No direct addition possible. Try a swap:
            # Remove an edge (a,b) where a or b is a high-degree vertex,
            # then try to add an edge to v.
            
            # Find high-degree vertices
            max_deg = max(degrees)
            high_verts = [i for i, d in enumerate(degrees) if d >= max_deg - 1]
            
            if not high_verts:
                break
            
            a = rng.choice(high_verts)
            
            # Find a neighbor b of a to remove
            neighbors_a = []
            mask = adj[a]
            while mask:
                lsb = mask & -mask
                b = lsb.bit_length() - 1
                neighbors_a.append(b)
                mask ^= lsb
            
            if not neighbors_a:
                break
            
            b = rng.choice(neighbors_a)
            
            # Save state before swap
            old_adj_a = adj[a]
            old_adj_b = adj[b]
            old_deg_a = degrees[a]
            old_deg_b = degrees[b]
            
            # Remove edge (a, b)
            adj[a] &= ~(1 << b)
            adj[b] &= ~(1 << a)
            degrees[a] -= 1
            degrees[b] -= 1
            
            # Try to add an edge to v
            non_neighbors_v = [u for u in range(n) if u != v and not (adj[v] & (1 << u))]
            rng.shuffle(non_neighbors_v)
            
            added_any = False
            added_u = -1
            for u in non_neighbors_v:
                if not _check_c4_bitset(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    degrees[u] += 1
                    degrees[v] += 1
                    added_any = True
                    added_u = u
                    break
            
            # Check if the swap improved things
            new_min = min(degrees)
            if new_min > min_deg:
                # Keep the swap
                pass
            else:
                # Revert the swap
                adj[a] = old_adj_a
                adj[b] = old_adj_b
                degrees[a] = old_deg_a
                degrees[b] = old_deg_b
                
                if added_any:
                    adj[added_u] &= ~(1 << v)
                    adj[v] &= ~(1 << added_u)
                    degrees[added_u] -= 1
                    degrees[v] -= 1
                
                # If we can't improve, try a different strategy
                # Just try a random restart with a different seed
                break
    
    # Final check
    if min(degrees) >= min_deg_req:
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    edges.append([u, v])
        return {'n': n, 'edges': edges}
    
    return None
