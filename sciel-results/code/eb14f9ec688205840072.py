"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 78  # Safety margin before 90s limit

    n = 16

    def _bfs_diameter(adj, n):
        """Compute diameter via BFS from each vertex. Returns (diameter, is_connected)."""
        diam = 0
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                u = queue[qi]
                qi += 1
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            if any(d == -1 for d in dist):
                return (n + 1, False)
            md = max(dist)
            if md > diam:
                diam = md
        return (diam, True)

    def _is_edge_critical_diam2(adj, n, edges):
        """Check if every single edge removal increases diameter above 2 or disconnects."""
        for (u, v) in edges:
            # Temporarily remove edge (u,v)
            adj[u].remove(v)
            adj[v].remove(u)
            d, connected = _bfs_diameter(adj, n)
            # Restore
            adj[u].append(v)
            adj[v].append(u)
            if d <= 2 and connected:
                return False
        return True

    def _build_adj(edges, n):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    # Start with K_{8,8} (64 edges) as baseline
    base_edges = []
    for u in range(8):
        for v in range(8, 16):
            base_edges.append((u, v))
    
    best_edges = [list(e) for e in base_edges]
    best_count = len(best_edges)

    # Try to add edges within partitions to create triangles
    # While maintaining edge-criticality for diameter 2
    # Strategy: try adding edges among left partition (0-7) or right partition (8-15)
    # and check if edge-criticality is preserved
    
    # First, verify K_{8,8} is valid
    adj = _build_adj(best_edges, n)
    d, connected = _bfs_diameter(adj, n)
    if d == 2 and connected:
        if _is_edge_critical_diam2(adj, n, best_edges):
            pass  # K_{8,8} confirmed valid with 64 edges
        else:
            # Should not happen, but fallback
            best_edges = []
            best_count = 0
    else:
        best_edges = []
        best_count = 0

    # Now try to add more edges
    # We'll try adding edges within left partition {0,...,7} and within right partition {8,...,15}
    left = list(range(8))
    right = list(range(8, 16))
    
    # Candidate edges to try adding: all intra-partition edges
    candidate_edges = []
    for u, v in itertools.combinations(left, 2):
        candidate_edges.append((u, v))
    for u, v in itertools.combinations(right, 2):
        candidate_edges.append((u, v))
    
    # Shuffle for diversity in which edges get tried first
    import random
    random.seed(42)
    random.shuffle(candidate_edges)
    
    # Greedily try to add edges that preserve edge-criticality
    current_edges = list(best_edges)
    adj = _build_adj(current_edges, n)
    
    for (cu, cv) in candidate_edges:
        if time.time() > deadline:
            break
        # Check if edge already exists
        if cv in adj[cu]:
            continue
        # Try adding this edge
        new_edges = current_edges + [(cu, cv)]
        new_adj = _build_adj(new_edges, n)
        
        # Check diameter is still 2
        d, connected = _bfs_diameter(new_adj, n)
        if d != 2 or not connected:
            continue
        
        # Check edge-criticality
        if _is_edge_critical_diam2(new_adj, n, new_edges):
            current_edges = new_edges
            adj = new_adj
            if len(current_edges) > best_count:
                best_count = len(current_edges)
                best_edges = [list(e) for e in current_edges]
        
        if time.time() > deadline:
            break

    # Ensure proper types: lists of lists of ints
    result = []
    for u, v in best_edges:
        result.append([int(u), int(v)])
    
    return result
# EVOLVE-BLOCK-END
