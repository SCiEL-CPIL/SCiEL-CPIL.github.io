import time
import random
import sys

# EVOLVE-BLOCK-START

def _to_bitmask(adj_list):
    """Convert list of neighbors to bitmask."""
    mask = 0
    for v in adj_list:
        mask |= (1 << v)
    return mask

def _find_k4(adj, n):
    """
    Find a K4 in the graph represented by adjacency bitmasks.
    adj[i] is an integer bitmask of neighbors of i.
    Returns (u, v, w, x) or None.
    """
    for u in range(n):
        # Only consider neighbors v > u to avoid duplicates
        neighbors_u = adj[u] & ~((1 << (u + 1)) - 1)
        v_iter = neighbors_u
        while v_iter:
            # Extract lowest set bit
            v_bit = v_iter & -v_iter
            v = v_bit.bit_length() - 1
            v_iter ^= v_bit
            
            # Check if u-v is an edge (it is, by construction)
            # Common neighbors of u and v, with index > v
            common = adj[u] & adj[v] & ~((1 << (v + 1)) - 1)
            
            # Check if common contains an edge
            w_iter = common
            while w_iter:
                w_bit = w_iter & -w_iter
                w = w_bit.bit_length() - 1
                w_iter ^= w_bit
                
                # Check if w-x is an edge for some x in common with x > w
                # Actually, we just need to check if there is any edge in 'common'
                # Let's check neighbors of w within 'common'
                # x must be in common and adj[w] must have x
                # x > w is not strictly required for validity, but we can restrict to x > w for uniqueness
                # However, adj[w] includes all neighbors.
                # We need x such that x is in common AND adj[w] has x.
                # And x > w to avoid duplicates? 
                # If we find ANY x, we have K4 {u, v, w, x}.
                # Since u < v < w, we just need x in common.
                # Wait, common are neighbors of u and v.
                # If w and x are in common, they are neighbors of u and v.
                # If w and x are neighbors of each other, {u,v,w,x} is K4.
                
                # Check if adj[w] intersects common (excluding w itself, but w is not in adj[w])
                # We need x in common such that adj[w] has x.
                # Let's just check if (adj[w] & common) != 0.
                # But we must ensure x != w. w is not in adj[w] so it's fine.
                # However, we should pick a specific x to return.
                
                potential_x = adj[w] & common
                if potential_x:
                    # Find one x
                    x_bit = potential_x & -potential_x
                    x = x_bit.bit_length() - 1
                    return (u, v, w, x)
                    
    return None

def _has_ind_in_set(mask, adj, n, size):
    """
    Check if the set of vertices defined by 'mask' contains an independent set of 'size'.
    Returns the tuple of vertices or None.
    """
    # Extract vertices from mask
    vertices = []
    m = mask
    while m:
        bit = m & -m
        v = bit.bit_length() - 1
        vertices.append(v)
        m ^= bit
    
    k = len(vertices)
    if k < size:
        return None
        
    # If size is small, brute force combinations
    # For size=4, k is usually < 20.
    if size == 4:
        # Use recursive or itertools
        from itertools import combinations
        for combo in combinations(vertices, 4):
            u, v, w, x = combo
            # Check if no edges
            if not (adj[u] & (1 << v)) and not (adj[u] & (1 << w)) and not (adj[u] & (1 << x)) and \
               not (adj[v] & (1 << w)) and not (adj[v] & (1 << x)) and not (adj[w] & (1 << x)):
                return (u, v, w, x)
    return None

def _find_ind6(adj, n):
    """
    Find an independent set of size 6.
    Returns tuple of 6 vertices or None.
    """
    for u in range(n):
        # Non-neighbors of u, with index > u
        non_neigh_u = ~adj[u] & ~((1 << (u + 1)) - 1) & ((1 << n) - 1)
        non_neigh_u &= ~(1 << u) # u is not neighbor of itself, but mask logic
        
        v_iter = non_neigh_u
        while v_iter:
            v_bit = v_iter & -v_iter
            v = v_bit.bit_length() - 1
            v_iter ^= v_bit
            
            # u and v are not connected.
            # Common non-neighbors of u and v, with index > v
            common_non = non_neigh_u & ~adj[v] & ~((1 << (v + 1)) - 1)
            common_non &= ~((1 << n) - 1)
            
            # We need an independent set of size 4 in common_non
            # Because {u, v} + {4 indep in common} = 6 indep
            # Note: u and v are not connected to anyone in common_non (by definition of common_non)
            # So we just need 4 vertices in common_non that are mutually non-adjacent.
            
            ind4 = _has_ind_in_set(common_non, adj, n, 4)
            if ind4:
                return (u, v) + ind4
    return None

def _is_valid(adj, n):
    return _find_k4(adj, n) is None and _find_ind6(adj, n) is None

def _repair_local_search(n, seed, max_steps=20000):
    """
    Perform local search to find a valid graph of size n.
    """
    random.seed(seed)
    
    # Initialize with random graph
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Check initial validity
    if _is_valid(adj, n):
        return adj
    
    for step in range(max_steps):
        # Find violation
        k4 = _find_k4(adj, n)
        if k4:
            # Remove a random edge from the K4
            u, v, w, x = k4
            edges = [(u, v), (u, w), (u, x), (v, w), (v, x), (w, x)]
            a, b = random.choice(edges)
            adj[a] &= ~(1 << b)
            adj[b] &= ~(1 << a)
            continue
        
        ind6 = _find_ind6(adj, n)
        if ind6:
            # Add a random edge to the I6
            vertices = ind6
            # Pick a random pair that is not connected
            # Actually, any pair in an independent set is not connected.
            # But we should check if adding the edge creates a K4.
            # For now, just add it. The next iteration will fix K4 if created.
            # To be smarter, pick a pair that minimizes new K4s?
            # Let's just pick random pair.
            a, b = random.sample(vertices, 2)
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
            continue
        
        # No violations found?
        # This shouldn't happen if _is_valid is correct, but just in case
        if _is_valid(adj, n):
            return adj
        
        # Escape local minima: Random flip
        # Add or remove a random edge
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        if adj[u] & (1 << v):
            # Remove
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            # Add
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_adj = None
    best_n = 1
    
    # Try from high n down to low n
    # We know 35 is possible.
    for n in range(35, 16, -1):
        if time.time() > deadline:
            break
            
        # Try multiple seeds for this n
        found = False
        for seed in range(100):
            if time.time() > deadline:
                break
                
            adj = _repair_local_search(n, seed, max_steps=5000)
            if adj is not None and _is_valid(adj, n):
                best_adj = adj
                best_n = n
                found = True
                break
        
        if found:
            break
    
    # If we didn't find anything, fall back to a known small graph
    if best_adj is None:
        # Paley 17
        n = 17
        squares = {x * x % 17 for x in range(1, 17)}
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % 17 in squares:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        best_adj = adj
        best_n = n
    
    # Convert to edge list
    edges = []
    for i in range(best_n):
        m = best_adj[i]
        while m:
            bit = m & -m
            j = bit.bit_length() - 1
            if i < j:
                edges.append([i, j])
            m ^= bit
            
    return {'n': best_n, 'edges': edges}

# EVOLVE-BLOCK-END
