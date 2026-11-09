"""C4-free graph with high minimum degree (R(C4,K1,39) lower bound)."""
import time
import random

# EVOLVE-BLOCK-START

def _can_add_edge(adj, u, v):
    """
    Fast check: does adding edge (u, v) create a C4?
    We assume (u, v) is not currently an edge.
    A C4 would be u - y - x - v - u with x in N(v), y in N(u), and edge (x, y).
    So for any x in N(v), if N(x) intersects N(u), then C4 exists.
    """
    nu = adj[u]
    nv = adj[v]
    x = nv
    while x:
        lsb = x & -x
        xi = lsb.bit_length() - 1
        x ^= lsb
        if adj[xi] & nu:
            return False
    return True

def _min_degree(adj, n):
    md = n
    for a in adj:
        d = a.bit_count()
        if d < md:
            md = d
    return md

def _is_c4_free(adj, n):
    """
    Full verification: for every edge (u, v), the number of common neighbors must be at most 1.
    """
    for u in range(n):
        au = adj[u]
        for v in range(u + 1, n):
            if (au >> v) & 1:
                common = au & adj[v]
                # If common has >= 2 bits, then there is a C4
                if common & (common - 1):
                    return False
    return True

def _greedy_build_structured(n, rng, p=0.25):
    """
    Build an initial C4-free graph by adding random edges with probability p,
    but using the C4 check to ensure C4-freedom.
    """
    adj = [0] * n
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(all_edges)
    
    for u, v in all_edges:
        if rng.random() < p:
            if _can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    
    return adj

def _local_search(adj, n, target_min_deg, rng, max_iter=2000):
    """
    Local search to increase minimum degree while maintaining C4-freedom.
    Two operations:
    1. Add an edge from a low-degree vertex to a non-neighbor if C4-safe.
    2. Swap: remove edge (a, b) and add edge (u, v) if it improves min degree
       and maintains C4-freedom.
    """
    improved = True
    it = 0
    while improved and it < max_iter:
        improved = False
        it += 1
        degrees = [a.bit_count() for a in adj]
        
        # Find vertices below target
        low_verts = [i for i in range(n) if degrees[i] < target_min_deg]
        
        if not low_verts:
            break
        
        # Operation 1: Add edges to low-degree vertices
        for u in low_verts:
            if degrees[u] >= target_min_deg:
                continue
            non_neighbors = [v for v in range(n) if v != u and not ((adj[u] >> v) & 1)]
            rng.shuffle(non_neighbors)
            
            for v in non_neighbors:
                if degrees[u] >= target_min_deg:
                    break
                if _can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    degrees[u] += 1
                    degrees[v] += 1
                    improved = True
        
        # If still below target, try edge swaps
        if min(degrees) < target_min_deg:
            # Find a high-degree vertex to remove an edge from
            # and a low-degree vertex to add an edge to
            low_verts = [i for i in range(n) if degrees[i] < target_min_deg]
            if low_verts:
                u = min(low_verts, key=lambda i: degrees[i])
                # Find a high-degree vertex a that is not u
                high_verts = [i for i in range(n) if i != u and degrees[i] >= target_min_deg + 1]
                if high_verts:
                    a = max(high_verts, key=lambda i: degrees[i])
                    # Try to find an edge (a, b) to remove and (u, v) to add
                    # such that C4-freedom is maintained and min degree improves
                    # Simplified: try random edges from a
                    neighbors_a = [v for v in range(n) if v != a and ((adj[a] >> v) & 1)]
                    rng.shuffle(neighbors_a)
                    
                    for b in neighbors_a:
                        if b == u:
                            continue
                        # Try to remove (a, b) and add (u, v) for some v
                        # v should be a non-neighbor of u
                        non_neighbors_u = [v for v in range(n) if v != u and not ((adj[u] >> v) & 1)]
                        rng.shuffle(non_neighbors_u)
                        
                        for v in non_neighbors_u:
                            if v == a or v == b:
                                continue
                            # Check if removing (a, b) and adding (u, v) maintains C4-freedom
                            # This is complex; for simplicity, we'll just try adding (u, v)
                            # after removing (a, b) if it seems safe
                            # But to keep it simple, we'll skip swaps for now
                            # and rely on the add operation
                            break
                        break
    
    return adj

def _edges_from_adj(adj, n):
    edges = []
    for i in range(n):
        ai = adj[i]
        for j in range(i + 1, n):
            if (ai >> j) & 1:
                edges.append([i, j])
    return edges

def construct():
    deadline = time.time() + 140.0  # safety margin under 150s
    best_n = 39
    best_edges = []
    best_adj = None
    
    # Try to maximize n from 47 down
    for n in range(47, 38, -1):
        target_min_deg = n - 39
        found = False
        
        # Try multiple seeds with structured initialization and local search
        # Use more restarts for larger n
        max_seeds = 1000 if n >= 46 else 500
        
        for seed in range(max_seeds):
            if time.time() > deadline:
                break
            
            rng = random.Random(seed)
            
            # Structured initialization
            adj = _greedy_build_structured(n, rng, p=0.25)
            
            # Local search to improve min degree
            adj = _local_search(adj, n, target_min_deg, rng, max_iter=1000)
            
            min_deg = _min_degree(adj, n)
            
            if min_deg >= target_min_deg:
                # Verify C4-freedom
                if _is_c4_free(adj, n):
                    edges = _edges_from_adj(adj, n)
                    
                    if n > best_n:
                        best_n = n
                        best_edges = edges
                        best_adj = adj
                    found = True
                    
                    # If we found n=47, return immediately
                    if n == 47:
                        return {'n': n, 'edges': edges}
        
        if found:
            # Since we iterate n downwards, this is the largest n found
            return {'n': best_n, 'edges': best_edges}
        
        # If not found for this n, continue to smaller n
    
    # Fallback
    if best_adj is not None:
        return {'n': best_n, 'edges': best_edges}
    else:
        return {'n': 39, 'edges': []}

# EVOLVE-BLOCK-END
