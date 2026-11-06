"""Rewritten program for the suite task; builds a C4-free graph via polarity graph + local search."""
import random
import sys

def _is_prime_power(q):
    """Check if q is a prime power."""
    if q < 2:
        return False
    # Trial division up to sqrt(q)
    i = 2
    while i * i <= q:
        if q % i == 0:
            # Check if q is a power of i
            temp = q
            while temp % i == 0:
                temp //= i
            if temp == 1:
                return True
            return False
        i += 1
    return True  # q is prime

def _build_polarity_graph(q):
    """
    Build the polarity graph of the projective plane of order q.
    Only works for prime q.
    n = q^2 + q + 1 vertices.
    Each vertex has degree q + 1.
    The graph is C4-free.
    """
    n = q * q + q + 1
    edges = []
    
    # Assign coordinates:
    # Type 1: (1, a, b) for a,b in Z_q -> q^2 points
    # Type 2: (0, 1, c) for c in Z_q -> q points
    # Type 3: (0, 0, 1) -> 1 point
    coords = []
    for a in range(q):
        for b in range(q):
            coords.append((1, a, b))
    for c in range(q):
        coords.append((0, 1, c))
    coords.append((0, 0, 1))
    
    # Build edges
    for i in range(n):
        x1, y1, z1 = coords[i]
        for j in range(i + 1, n):
            x2, y2, z2 = coords[j]
            if (x1 * x2 + y1 * y2 + z1 * z2) % q == 0:
                edges.append([i, j])
    
    return n, edges

def _has_c4(n, adj):
    """Check if graph has a C4. adj is a list of sets."""
    for i in range(n):
        for j in adj[i]:
            if j <= i:
                continue
            # Check if i and j have 2 common neighbors
            common = adj[i] & adj[j]
            if len(common) >= 2:
                return True
    return False

def _min_degree(n, adj):
    return min(len(adj[i]) for i in range(n))

def _try_local_search(n_target, min_deg_needed, adj, edges, rng, max_iters=50000):
    """
    Apply local search to add edges while maintaining C4-free property.
    Returns (success, adj, edges)
    """
    for _ in range(max_iters):
        # Check if we've achieved the target
        current_min_deg = min(len(adj[i]) for i in range(n_target))
        if current_min_deg >= min_deg_needed:
            return True, adj, edges
        
        # Find vertices with degree below target
        low_deg_vertices = [i for i in range(n_target) if len(adj[i]) < min_deg_needed]
        if not low_deg_vertices:
            return True, adj, edges
        
        rng.shuffle(low_deg_vertices)
        
        added = False
        for u in low_deg_vertices:
            if len(adj[u]) >= min_deg_needed:
                continue
            
            # Find candidates v such that adding (u,v) doesn't create C4
            # Condition: |N(u) ∩ N(v)| < 2
            candidates = []
            for v in range(n_target):
                if v == u or v in adj[u]:
                    continue
                common = adj[u] & adj[v]
                if len(common) < 2:
                    candidates.append(v)
            
            if candidates:
                v = rng.choice(candidates)
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v])
                added = True
                break
        
        if not added:
            break
    
    # Final check
    if min(len(adj[i]) for i in range(n_target)) >= min_deg_needed:
        return True, adj, edges
    return False, adj, edges

def construct():
    """
    Construct a C4-free graph with maximum n such that min degree >= n - 39.
    """
    # Try n = 47 first
    for n_target in [47, 46, 45, 44, 43]:
        min_deg_needed = n_target - 39
        
        if min_deg_needed <= 0:
            # Trivially satisfied, but we want to maximize n
            # Still need to ensure C4-free
            # Use polarity graph of q=5 (n=31) as core, add isolated vertices
            q = 5
            n_core, core_edges = _build_polarity_graph(q)
            adj = [set() for _ in range(n_target)]
            edges = []
            for u, v in core_edges:
                if u < n_target and v < n_target:
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.append([u, v])
            
            # Add edges between new vertices if possible
            rng = random.Random(n_target * 1000 + 42)
            success, adj, edges = _try_local_search(n_target, min_deg_needed, adj, edges, rng)
            
            if success and not _has_c4(n_target, adj):
                return {'n': n_target, 'edges': edges}
            continue
        
        # Use polarity graph of q=5 (n=31) as core
        q = 5
        n_core, core_edges = _build_polarity_graph(q)
        
        # Initialize adjacency
        adj = [set() for _ in range(n_target)]
        edges = []
        for u, v in core_edges:
            if u < n_target and v < n_target:
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v])
        
        # Try multiple random seeds for local search
        for seed in range(20):
            rng = random.Random(seed * 10000 + n_target)
            # Make a copy
            adj_copy = [set(s) for s in adj]
            edges_copy = [e[:] for e in edges]
            
            success, adj_copy, edges_copy = _try_local_search(n_target, min_deg_needed, adj_copy, edges_copy, rng, max_iters=10000)
            
            if success and not _has_c4(n_target, adj_copy):
                return {'n': n_target, 'edges': edges_copy}
        
        # If all seeds failed, try a different approach: start with more random edges
        for seed in range(10):
            rng = random.Random(seed * 20000 + n_target)
            adj = [set() for _ in range(n_target)]
            edges = []
            
            # Start with polarity graph core
            for u, v in core_edges:
                if u < n_target and v < n_target:
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.append([u, v])
            
            # Add some random edges initially
            for _ in range(50):
                u = rng.randint(0, n_target - 1)
                v = rng.randint(0, n_target - 1)
                if u != v and v not in adj[u]:
                    common = adj[u] & adj[v]
                    if len(common) < 2:
                        adj[u].add(v)
                        adj[v].add(u)
                        edges.append([u, v])
            
            success, adj, edges = _try_local_search(n_target, min_deg_needed, adj, edges, rng, max_iters=20000)
            
            if success and not _has_c4(n_target, adj):
                return {'n': n_target, 'edges': edges}
    
    # Fallback: return polarity graph of q=5
    n_core, core_edges = _build_polarity_graph(5)
    return {'n': n_core, 'edges': core_edges}
