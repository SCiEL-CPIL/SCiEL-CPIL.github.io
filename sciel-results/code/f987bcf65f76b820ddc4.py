import time
import random
import numpy as np

def _build_adj(n, edges):
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _has_k4(adj, n):
    """Check for K4 in graph with adjacency list (bitmasks)."""
    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] >> v) & 1:
                common = adj[u] & adj[v]
                m = common
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    m ^= lsb
                    if w > v and (common & adj[w]):
                        # Found triangle u,v,w; check if any common neighbor of all three
                        tri_common = adj[u] & adj[v] & adj[w]
                        if tri_common:
                            return True
    return False

def _has_independent_6(adj, n):
    """Check for independent set of size 6 (clique of size 6 in complement)."""
    comp_adj = [0] * n
    full_mask = (1 << n) - 1
    for i in range(n):
        comp_adj[i] = full_mask ^ adj[i] ^ (1 << i)
    
    def dfs(verts, depth):
        if depth == 6:
            return True
        if not verts:
            return False
        v = verts
        while v:
            lsb = v & -v
            u = lsb.bit_length() - 1
            v ^= lsb
            new_verts = v & comp_adj[u]
            if dfs(new_verts, depth + 1):
                return True
    return dfs(full_mask, 0)

def _is_valid_graph(n, adj):
    if _has_k4(adj, n):
        return False
    if _has_independent_6(adj, n):
        return False
    return True

def _edges_from_adj(adj, n):
    edges = []
    for u in range(n):
        m = adj[u]
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            if u < v:
                edges.append([u, v])
    return edges

def _find_exoo_35_graph():
    """
    Return the known 35-vertex K4-free graph with alpha < 6 (alpha=5).
    This is based on the exoo graph for R(4,6). We construct it via
    the known construction: take the 35-vertex graph from the literature.
    
    A well-known 35-vertex example can be generated from the Paley graph
    of order 37 with 2 vertices removed, or from other constructions.
    
    Actually, let's use a more reliable approach: we'll use a known valid
    35-vertex graph. The exoo graph for R(4,6) > 35 is well-documented.
    
    For now, let's construct a valid graph using a different method:
    We'll use the known fact that the 35-vertex graph exists and try
    to generate it via a known construction.
    
    A reliable source: the graph can be constructed as the complement of
    a (35, 7, 3) design or similar. But let's use a simpler approach:
    we'll hardcode a known valid 35-vertex graph's edge list, or generate
    one via a verified construction.
    
    Actually, let's use a different strategy: we'll start from a known
    valid smaller graph and try to extend it. But the key insight from
    the cell findings is that starting from the known 35-vertex graph
    is most promising.
    
    Let me use a known construction: the 35-vertex graph can be obtained
    from the Paley graph of order 37 by deleting two non-adjacent vertices
    that form an independent set of size 2 in the Paley graph.
    
    Wait, Paley(37) has 37 vertices. If we remove 2 vertices, we get 35.
    But we need to ensure the result is still K4-free and has no independent
    6-set.
    
    Let me try a different approach: I'll use a known valid 35-vertex graph
    by constructing it from a (35, 5, 1) design or similar structure.
    
    For robustness, let me try to generate a valid graph using a heuristic
    approach with a fixed seed, and if it works, use it.
    """
    # Let's try to construct a valid 35-vertex graph using a known method.
    # The exoo graph for R(4,6) > 35 is well-known. Let me try to generate
    # it using a systematic approach.
    
    # One known construction: take the 5x7 grid of vertices (35 vertices)
    # and define edges based on a specific pattern.
    
    # Actually, let me use a more direct approach: I'll try to generate
    # a valid graph using random initialization with a fixed seed and
    # local search to fix violations.
    
    n = 35
    rng = random.Random(42)
    
    # Initialize with a random graph
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Local search to fix K4 and independent 6 violations
    for iteration in range(10000):
        if not _has_k4(adj, n) and not _has_independent_6(adj, n):
            break
        
        # Find a K4 and break it
        k4_found = False
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    m = common
                    while m:
                        lsb = m & -m
                        w = lsb.bit_length() - 1
                        m ^= lsb
                        if w > v and (common & adj[w]):
                            tri_common = adj[u] & adj[v] & adj[w]
                            if tri_common:
                                # Found K4: u, v, w, x where x is in tri_common
                                x_lsb = tri_common & -tri_common
                                x = x_lsb.bit_length() - 1
                                # Remove one edge to break the K4
                                adj[u] &= ~(1 << x)
                                adj[x] &= ~(1 << u)
                                k4_found = True
                                break
                    if k4_found:
                        break
            if k4_found:
                break
        
        if k4_found:
            continue
        
        # Find an independent 6-set and break it
        if _has_independent_6(adj, n):
            # Add a random edge to reduce independence
            u = rng.randint(0, n - 1)
            v = rng.randint(0, n - 1)
            if u != v and not ((adj[u] >> v) & 1):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    
    if _is_valid_graph(n, adj):
        return n, adj
    
    # If the random approach didn't work, return None
    return None, None

def _try_extend_to_36(adj35, deadline):
    """Try to extend a valid 35-vertex graph to 36 vertices."""
    n = 36
    start_time = time.time()
    
    # Add a new vertex 35
    adj = adj35[:] + [0]
    
    # Connect the new vertex to some existing vertices
    # Strategy: try different connection patterns
    for attempt in range(1000):
        if time.time() > deadline:
            break
        
        # Randomly connect vertex 35 to some vertices
        for i in range(35):
            if random.random() < 0.5:
                adj[35] |= (1 << i)
                adj[i] |= (1 << 35)
            else:
                adj[35] &= ~(1 << i)
                adj[i] &= ~(1 << 35)
        
        # Check validity
        if _is_valid_graph(n, adj):
            return adj
        
        # If invalid, try to fix by flipping edges incident to vertex 35
        # or by flipping other edges
        for fix_attempt in range(100):
            if time.time() > deadline:
                break
            
            if not _is_valid_graph(n, adj):
                # Try flipping a random edge
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u != v:
                    # Flip edge (u, v)
                    if (adj[u] >> v) & 1:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                    
                    if _is_valid_graph(n, adj):
                        return adj
    
    return None

def _paley_graph(p):
    """Generate Paley graph for prime p."""
    squares = {x * x % p for x in range(1, p)}
    edges = []
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                edges.append([i, j])
    return {'n': p, 'edges': edges}

def _clebsch_graph():
    """Generate the Clebsch graph: 16 vertices, K4-free, alpha=5."""
    n = 16
    vertices = []
    for i in range(16):
        bits = bin(i).count('1')
        if bits % 2 == 0:
            vertices.append(i)
    idx_map = {v: i for i, v in enumerate(vertices)}
    edges = []
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            diff = bin(vertices[i] ^ vertices[j]).count('1')
            if diff in (1, 3):
                edges.append([i, j])
    return {'n': n, 'edges': edges}

def _try_paley_candidates(deadline):
    """Try Paley graphs as fallback."""
    candidates = [37, 29, 17]
    best_graph = None
    best_n = 0
    
    for p in candidates:
        if time.time() > deadline:
            break
        g = _paley_graph(p)
        adj = _build_adj(g['n'], g['edges'])
        if _is_valid_graph(g['n'], adj):
            if g['n'] > best_n:
                best_graph = g
                best_n = g['n']
    
    return best_graph, best_n

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # First, try to find a valid 35-vertex graph
    n35, adj35 = _find_exoo_35_graph()
    
    # If we found a valid 35-vertex graph, try to extend to 36
    if n35 == 35 and adj35 is not None:
        if time.time() < deadline:
            adj36 = _try_extend_to_36(adj35, deadline)
            if adj36 is not None:
                edges = _edges_from_adj(adj36, 36)
                return {'n': 36, 'edges': edges}
    
    # Fallback: try Paley graphs
    best_graph, best_n = _try_paley_candidates(deadline)
    
    # Fallback: Clebsch graph
    if time.time() < deadline:
        g = _clebsch_graph()
        adj = _build_adj(g['n'], g['edges'])
        if _is_valid_graph(g['n'], adj):
            if g['n'] > best_n:
                best_graph = g
                best_n = g['n']
    
    if best_graph is None:
        best_graph = _clebsch_graph()
    
    return best_graph
