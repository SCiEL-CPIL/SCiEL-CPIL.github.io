"""C4-free graph with high minimum degree (R(C4,K1,39) lower bound)."""
import time
import random

# EVOLVE-BLOCK-START

def _can_add_edge(adj, n, u, v):
    """
    Check if adding edge (u, v) creates a C4.
    A C4 is formed if there exist x, y such that:
    u-x, x-v, v-y, y-u are edges.
    This implies x is a neighbor of v (in current graph) and y is a neighbor of u (in current graph).
    And (x, y) is an edge in the current graph.
    Also x != y (to avoid triangle u-x-v-y-u where x=y? No, if x=y, it's triangle u-x-v. Triangles are OK).
    So we check if there is an edge between N(u) and N(v) connecting distinct vertices.
    """
    # Get neighbors of u and v
    nu = adj[u]
    nv = adj[v]
    
    # Iterate through neighbors of u
    # For each y in N(u), check if it is connected to any x in N(v) where x != y
    # We can optimize by checking neighbors of v
    # Let's iterate over x in N(v)
    x = nv
    while x:
        lsb = x & -x
        xi = lsb.bit_length() - 1
        x ^= lsb
        
        if xi == u or xi == v:
            continue
            
        # Check if xi is connected to any y in N(u) where y != xi
        # We need to check if N(xi) intersects N(u) excluding xi
        # N(xi) is adj[xi]
        # N(u) is nu
        # Intersection: adj[xi] & nu
        # If intersection has a bit set that is not xi, then C4 exists.
        # Note: xi is not in nu (simple graph, no self loops, u-xi edge exists so xi in nu? 
        # Wait. If u-xi is an edge, then xi is in N(u). 
        # But we are checking if x (which is xi) is connected to y in N(u).
        # If xi is in N(u), then u-xi is an edge.
        # The C4 would be u - xi - v - y - u.
        # This requires v-y and y-u.
        # So y must be in N(v) and N(u).
        # And xi must be in N(v).
        # And edge (xi, y) must exist.
        
        # So:
        # 1. xi is in N(v) (true by loop)
        # 2. We need y in N(u) such that (xi, y) is an edge.
        # 3. y != xi.
        
        # Let's check common neighbors of xi and u.
        # common = adj[xi] & nu
        # If common has any bit set, say y.
        # If y == xi, then xi is in N(u) and N(xi)? No, xi not in N(xi).
        # So y cannot be xi.
        # So if adj[xi] & nu is non-zero, there is a y in N(u) connected to xi.
        # This forms u-y-xi-v-u? 
        # Edges: u-y (y in N(u)), y-xi (checked), xi-v (xi in N(v)), v-u (adding).
        # Yes, this is a C4.
        
        if adj[xi] & nu:
            return False
            
    return True

def _greedy_build(n, rng):
    adj = [0] * n
    edges_to_try = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(edges_to_try)
    
    for u, v in edges_to_try:
        if _can_add_edge(adj, n, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
    # Calculate degrees
    degrees = [a.bit_count() for a in adj]
    min_deg = min(degrees)
    
    return adj, min_deg

def construct():
    deadline = time.time() + 135.0
    best_n = 39
    best_edges = []
    
    # Try to maximize n
    for n in range(47, 38, -1):
        target_min_deg = n - 39
        found = False
        # Try multiple seeds
        for seed in range(200):
            if time.time() > deadline:
                break
            rng = random.Random(seed)
            adj, min_deg = _greedy_build(n, rng)
            
            if min_deg >= target_min_deg:
                # Construct edge list
                edges = []
                for i in range(n):
                    ai = adj[i]
                    for j in range(i + 1, n):
                        if (ai >> j) & 1:
                            edges.append([i, j])
                
                if n > best_n:
                    best_n = n
                    best_edges = edges
                found = True
                break # Found valid graph for this n, try smaller n? No, we want max n.
                # So if we found n=47, we can return immediately?
                # Yes, 47 is the max possible.
                if n == 47:
                    return {'n': n, 'edges': edges}
        
        if found:
            # We found a valid graph for this n. Since we iterate n downwards, 
            # this is the largest n we found in this pass.
            # But we should check if we found a *better* min degree for the same n?
            # The task is to maximize n. So any valid n is better than smaller n.
            # So we can return the first valid n found (which is the largest).
            return {'n': n, 'edges': best_edges}
            
    # Fallback
    return {'n': best_n, 'edges': best_edges}

# EVOLVE-BLOCK-END
