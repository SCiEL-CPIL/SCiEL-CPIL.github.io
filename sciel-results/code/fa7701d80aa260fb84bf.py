"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

DEADLINE = 85  # seconds, safety margin before 90s kill

def _build_adjacency(edges, n):
    """Build adjacency list and set from edge list."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _diameter(adj, n):
    """Compute diameter using BFS from each vertex. Returns (diameter, is_connected)."""
    max_dist = 0
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        queue = deque([src])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    max_dist = max(max_dist, dist[v])
                    queue.append(v)
        if dist[-1] == -1 or any(d == -1 for d in dist):
            return -1, False  # disconnected
    return max_dist, True

def _check_all_edges_critical(adj, n, edges):
    """Check that removing each edge either disconnects or makes diameter > 2."""
    edge_set = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        edge_set.add((u, v))
    
    for (u, v) in edge_set:
        # Temporarily remove edge (u, v)
        adj[u].discard(v)
        adj[v].discard(u)
        diam, connected = _diameter(adj, n)
        # Restore
        adj[u].add(v)
        adj[v].add(u)
        if connected and diam <= 2:
            return False
    return True

def _is_valid(adj, n, edges):
    """Check diameter is exactly 2 and all edges are critical."""
    diam, connected = _diameter(adj, n)
    if not connected or diam != 2:
        return False
    return _check_all_edges_critical(adj, n, edges)

def _make_initial_graph(n):
    """Create an initial graph: K_n minus a perfect matching (if n even)."""
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (i + j) % 2 == 0 and i != j:  # This isn't quite a matching
                continue
            edges.append([i, j])
    # Actually, let's do K_n minus a specific structure
    # Let's just start with K_n minus a cycle
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            edges.append([i, j])
    # Remove a Hamiltonian cycle's edges
    cycle_edges = set()
    for i in range(n):
        j = (i + 1) % n
        a, b = min(i, j), max(i, j)
        cycle_edges.add((a, b))
    edges = [[u, v] for u, v in edges if (min(u,v), max(u,v)) not in cycle_edges]
    return edges

def construct():
    start_time = time.time()
    n = 16
    
    # Start with a moderately dense graph and try to optimize
    # Strategy: Start with K_n minus some edges, verify, then try to add edges back
    
    # Initial: complete graph minus a 2-regular spanning subgraph (union of cycles)
    # Let's try: K_16 minus a 16-cycle
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            edges.append([i, j])
    
    # Remove a Hamiltonian cycle
    cycle = list(range(n))
    cycle_edges = set()
    for i in range(n):
        a, b = cycle[i], cycle[(i+1) % n]
        cycle_edges.add((min(a,b), max(a,b)))
    edges = [[u, v] for u, v in edges if (min(u,v), max(u,v)) not in cycle_edges]
    
    adj = _build_adjacency(edges, n)
    
    # Check if valid
    if _is_valid(adj, n, edges):
        return edges
    
    # If not valid, try different removals
    # Try removing different structures
    # Let's try: for each possible "critical structure", build graph
    
    # Alternative: use a known construction
    # Consider: two cliques of 8 vertices, with cross edges forming a complete bipartite minus something
    
    # Let me try a different approach: start sparse and add edges
    # Start with a path or cycle, then add edges greedily
    
    # Greedy: start with empty, add edges that maintain validity
    # But this is slow. Let's try to find a good construction analytically.
    
    # Key insight: For max edges with all critical, consider graphs where
    # the "criticality" comes from each edge being the unique short path for some pair.
    
    # Let's try: the graph is K_n minus a matching, but that's not critical.
    # What about K_n minus a star? No, that disconnects.
    
    # Let me try a computational search with time limit
    import random
    
    best_edges = None
    best_count = -1
    
    # Try several constructions
    constructions = []
    
    # Construction 1: K_n minus Hamiltonian cycle
    e1 = [[i, j] for i in range(n) for j in range(i+1, n)]
    ce = set()
    for i in range(n):
        a, b = i, (i+1) % n
        ce.add((min(a,b), max(a,b)))
    e1 = [[u,v] for u,v in e1 if (min(u,v),max(u,v)) not in ce]
    constructions.append(e1)
    
    # Construction 2: K_n minus two disjoint cycles
    e2 = [[i, j] for i in range(n) for j in range(i+1, n)]
    ce2 = set()
    for i in range(8):
        a, b = i, (i+1) % 8
        ce2.add((min(a,b), max(a,b)))
    for i in range(8, 16):
        a, b = i, (i+1) % 8 + 8
        ce2.add((min(a,b), max(a,b)))
    e2 = [[u,v] for u,v in e2 if (min(u,v),max(u,v)) not in ce2]
    constructions.append(e2)
    
    # Construction 3: K_n minus a perfect matching
    e3 = [[i, j] for i in range(n) for j in range(i+1, n)]
    me = set()
    for i in range(0, n, 2):
        me.add((i, i+1))
    e3 = [[u,v] for u,v in e3 if (min(u,v),max(u,v)) not in me]
    constructions.append(e3)
    
    # Check each construction
    for e in constructions:
        adj = _build_adjacency(e, n)
        if _is_valid(adj, n, e):
            if len(e) > best_count:
                best_count = len(e)
                best_edges = e
            if time.time() - start_time > DEADLINE:
                break
    
    if best_edges is not None:
        return best_edges
    
    # Fallback: try a greedy approach with time limit
    # Start with a sparse valid graph and add edges
    # First, find ANY valid graph
    # Try: a "double star" or similar
    
    # Let's try: vertex 0 connected to all, vertices 1-15 form a structure
    # Actually, let's just return the best we have or a simple valid graph
    
    # Simple valid graph: a path of length 2 between all pairs via a hub
    # Star graph: 0 connected to 1-15. Diameter 2. But edges are not all critical
    # (removing an edge disconnects, so they ARE critical in that sense)
    # Wait - if removing an edge disconnects, that counts as "increases diameter above 2 or disconnects"
    # So a star IS valid!
    
    star_edges = [[0, v] for v in range(1, n)]
    adj = _build_adjacency(star_edges, n)
    if _is_valid(adj, n, star_edges):
        if star_edges is None or len(star_edges) > best_count:
            best_edges = star_edges
            best_count = len(star_edges)
    
    if best_edges is not None:
        return best_edges
    
    # Absolute fallback
    return [[0, v] for v in range(1, n)]
# EVOLVE-BLOCK-END
