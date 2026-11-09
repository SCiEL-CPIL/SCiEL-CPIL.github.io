"""C4-free graph with high minimum degree (R(C4,K1,39) lower bound)."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _can_add_edge(adj, u, v):
    """
    Check if adding edge (u, v) creates a C4.
    Returns True if (u, v) can be added without creating a C4.
    """
    nu = adj[u]
    nv = adj[v]
    
    # Iterate over neighbors of v
    x = nv
    while x:
        lsb = x & -x
        xi = lsb.bit_length() - 1
        x ^= lsb
        
        # Check if N(xi) intersects N(u)
        # If so, there is a y in N(u) such that y-x is an edge.
        # Then u-y-x-v-u is a C4.
        if adj[xi] & nu:
            return False
            
    return True

def _has_c4(adj, n):
    """
    Check if the graph contains any C4.
    Returns a list of edges that are part of a C4, or empty list.
    """
    c4_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] >> v) & 1:
                # Check common neighbors
                common = adj[u] & adj[v]
                # If there are at least 2 common neighbors, check if they are connected
                if common & (common - 1): # at least 2 bits
                    # Iterate pairs of common neighbors
                    c = common
                    while c:
                        lsb1 = c & -c
                        x = lsb1.bit_length() - 1
                        c ^= lsb1
                        # Check other common neighbors
                        c2 = common
                        while c2:
                            lsb2 = c2 & -c2
                            y = lsb2.bit_length() - 1
                            c2 ^= lsb2
                            if x < y:
                                if (adj[x] >> y) & 1:
                                    # C4: u-x-v-y-u
                                    c4_edges.append((u, x))
                                    c4_edges.append((x, v))
                                    c4_edges.append((v, y))
                                    c4_edges.append((y, u))
    return c4_edges

def _remove_c4s(adj, n, rng):
    """
    Remove edges to break all C4s.
    """
    improved = True
    while improved:
        improved = False
        c4_edges = _has_c4(adj, n)
        if c4_edges:
            # Pick a random edge from the C4s and remove it
            u, v = rng.choice(c4_edges)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            improved = True

def _edges_from_adj(adj, n):
    edges = []
    for i in range(n):
        ai = adj[i]
        for j in range(i + 1, n):
            if (ai >> j) & 1:
                edges.append([i, j])
    return edges

def _local_search(adj, n, target_min_deg, rng, deadline):
    """
    Local search to improve minimum degree.
    """
    best_min_deg = min(a.bit_count() for a in adj)
    
    while time.time() < deadline:
        degrees = [a.bit_count() for a in adj]
        min_deg = min(degrees)
        
        if min_deg > best_min_deg:
            best_min_deg = min_deg
        
        if min_deg >= target_min_deg:
            return adj, min_deg
            
        # Find a vertex with min degree
        min_verts = [i for i in range(n) if degrees[i] == min_deg]
        u = rng.choice(min_verts)
        
        # Try to add an edge to u
        non_neighbors = [v for v in range(n) if v != u and not (adj[u] >> v) & 1]
        rng.shuffle(non_neighbors)
        
        added = False
        for v in non_neighbors:
            if _can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                added = True
                break
        
        if not added:
            # Try to remove an edge to make room
            # Pick a random edge not incident to u
            edges = []
            for i in range(n):
                if i == u: continue
                ai = adj[i]
                for j in range(i + 1, n):
                    if j == u: continue
                    if (ai >> j) & 1:
                        edges.append((i, j))
            
            if edges:
                # Prefer removing edges from vertices with high degree
                edges.sort(key=lambda e: degrees[e[0]] + degrees[e[1]], reverse=True)
                # Try top few candidates
                for i, j in edges[:5]:
                    # Check if removing (i,j) allows adding (u,v) for some v
                    # Temporarily remove
                    adj[i] &= ~(1 << j)
                    adj[j] &= ~(1 << i)
                    
                    # Try adding (u, v) again
                    added_after_remove = False
                    for v in non_neighbors:
                        if _can_add_edge(adj, u, v):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            added_after_remove = True
                            break
                    
                    if not added_after_remove:
                        # Restore edge
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)

def construct():
    deadline = time.time() + 145.0  # safety margin
    best_n = 39
    best_edges = []
    
    # Try to maximize n from 47 down
    for n in range(47, 38, -1):
        target_min_deg = n - 39
        found = False
        
        # Try multiple seeds
        for seed in range(2000):
            if time.time() > deadline:
                break
            
            rng = random.Random(seed)
            
            # Initialize with random graph
            p = (n - 39) / (n - 1) * 1.2 # Slightly higher probability to start
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() < p:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Remove C4s
            _remove_c4s(adj, n, rng)
            
            # Local search to improve min degree
            adj, min_deg = _local_search(adj, n, target_min_deg, rng, deadline)
            
            if min_deg >= target_min_deg:
                edges = _edges_from_adj(adj, n)
                
                if n > best_n:
                    best_n = n
                    best_edges = edges
                found = True
                
                if n == 47:
                    return {'n': n, 'edges': edges}
        
        if found:
            return {'n': best_n, 'edges': best_edges}
    
    # Fallback
    if best_edges:
        return {'n': best_n, 'edges': best_edges}
    else:
        return {'n': 39, 'edges': []}

# EVOLVE-BLOCK-END
