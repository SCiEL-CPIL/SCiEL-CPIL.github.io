"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _has_triangle(n, adj, edge_set):
    # Check if adding edge (u,v) creates a triangle
    # A triangle exists if there's a vertex w adjacent to both u and v
    u, v = edge_set
    common = adj[u] & adj[v]
    return len(common) > 0

def _bfs_diameter(n, adj):
    # Compute diameter via BFS from each node
    from collections import deque
    max_dist = 0
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        max_dist = max(max_dist, max(dist))
    return max_dist

def _is_valid(n, edges):
    adj = _build_adj(n, edges)
    # Check connected
    from collections import deque
    dist = [-1] * n
    dist[0] = 0
    q = deque([0])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    if any(d == -1 for d in dist):
        return False
    # Check triangle-free
    edge_set = set()
    for u, v in edges:
        edge_set.add((u, v))
        edge_set.add((v, u))
    for u in range(n):
        for v in adj[u]:
            if u < v:
                for w in adj[v]:
                    if w > v and w in adj[u]:
                        return False
    # Check diameter
    diam = _bfs_diameter(n, adj)
    return diam <= 4

def _try_remove_edge(n, base_edges, added_edges):
    """Try to remove each added edge and check validity."""
    best = list(added_edges)
    improved = True
    while improved:
        improved = False
        for i in range(len(best)):
            candidate = best[:i] + best[i+1:]
            all_edges = base_edges + candidate
            if _is_valid(n, all_edges):
                best = candidate
                improved = True
                break
    return best

def construct():
    deadline = time.time() + 80  # 80s safety margin
    
    n = P['n']
    base_edges = P['edges']
    base_edge_set = set()
    for e in base_edges:
        base_edge_set.add((e[0], e[1]))
        base_edge_set.add((e[1], e[0]))
    
    # Generate candidate edges: all non-base edges that don't immediately create triangles with base
    # But first, let's build a good initial set
    # Strategy: add edges that connect vertices at distance 5-7 along the path
    # to create shortcuts
    
    # Initial guess: add edges to create a structure with diameter <= 4
    # Let's try a systematic approach: for each pair (u,v) where |u-v| >= 5,
    # consider adding edge if it doesn't create a triangle
    
    # First, build adjacency of base
    adj = _build_adj(n, base_edges)
    
    # Generate all possible edges not in base
    all_possible = []
    for u in range(n):
        for v in range(u+1, n):
            if (u, v) not in base_edge_set and (v, u) not in base_edge_set:
                all_possible.append((u, v))
    
    # Filter: edges that don't create triangles with the base graph
    valid_candidates = []
    for u, v in all_possible:
        common = adj[u] & adj[v]
        if len(common) == 0:
            valid_candidates.append((u, v))
    
    # Now, build initial set greedily: add edges that most reduce diameter
    # Start with empty added set
    added = []
    added_set = set()
    current_adj = _build_adj(n, base_edges)
    
    # Greedy: repeatedly add the edge that most reduces the current diameter
    # But computing diameter each time is expensive. Let's use a heuristic:
    # Add edges that connect far-apart vertices
    
    # Alternative: start with a known-good set and remove
    # Let's try: add edges (0,5), (1,6), (2,7), (3,8), (4,9), (5,10)
    # Actually, let's be smarter.
    
    # Try a set of edges that creates a "5-step skip" pattern
    # For diameter 4 on 24 nodes, we need roughly 24/5 = ~5-6 skip edges
    
    # Let's try multiple initial configurations and pick the best after removal
    
    initial_configs = []
    
    # Config 1: skip-5 pattern
    config1 = [(i, i+5) for i in range(0, 19, 2)]  # (0,5),(2,7),(4,9),(6,11),(8,13),(10,15),(12,17),(14,19),(16,21),(18,23)
    # Filter out invalid ones (triangles with base or each other)
    
    # Config 2: skip-6 pattern
    config2 = [(i, i+6) for i in range(0, 18, 2)]
    
    # Config 3: mixed
    config3 = [(0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11)]
    
    # Config 4: star-like from center
    config4 = [(12, 0), (12, 23), (6, 0), (18, 23)]
    
    # Let's just do a greedy construction properly
    # Start with all valid candidates, greedily add edges that help
    
    # Actually, the knowledge says greedy + removal finds 6 edges. Let me implement that.
    
    # Step 1: Build a superset that definitely works
    # Add all edges at distance 5 or 6 that are triangle-free with base
    superset = []
    current_adj = _build_adj(n, base_edges)
    current_edge_set = set(base_edge_set)
    
    # Sort candidates by "usefulness" - prefer edges that connect far-apart nodes
    # and are in the middle of the path
    def edge_usefulness(u, v):
        dist = abs(u - v)
        mid = (u + v) / 2
        # Prefer edges that are in the central region and have moderate distance
        return (dist, -abs(mid - 11.5))
    
    valid_candidates.sort(key=lambda e: (-abs(e[0]-e[1]), abs((e[0]+e[1])/2 - 11.5)))
    
    # Add edges greedily
    for u, v in valid_candidates:
        if time.time() > deadline:
            break
        # Check if adding this edge creates a triangle with current graph
        common = current_adj[u] & current_adj[v]
        if len(common) == 0:
            # Add it
            superset.append((u, v))
            current_adj[u].add(v)
            current_adj[v].add(u)
            current_edge_set.add((u, v))
            current_edge_set.add((v, u))
    
    # Now we have a superset. Check if it's valid
    all_edges = base_edges + superset
    if not _is_valid(n, all_edges):
        # Fallback: try a smaller, known-good configuration
        # Use the skip-5 pattern
        superset = []
        current_adj = _build_adj(n, base_edges)
        for i in range(0, n-5):
            u, v = i, i+5
            common = current_adj[u] & current_adj[v]
            if len(common) == 0:
                superset.append((u, v))
                current_adj[u].add(v)
                current_adj[v].add(u)
        all_edges = base_edges + superset
        if not _is_valid(n, all_edges):
            # Last resort: just return a minimal valid set
            # Try (0,5), (5,10), (10,15), (15,20), (20,23), (0,23)
            test_edges = [(0,5), (5,10), (10,15), (15,20), (20,23), (0,23)]
            superset = []
            current_adj = _build_adj(n, base_edges)
            for u, v in test_edges:
                common = current_adj[u] & current_adj[v]
                if len(common) == 0:
                    superset.append((u, v))
                    current_adj[u].add(v)
                    current_adj[v].add(u)
            all_edges = base_edges + superset
    
    # Step 2: Remove edges greedily
    result = _try_remove_edge(n, base_edges, superset)
    
    # Convert to list of lists
    return [[u, v] for u, v in result]

# EVOLVE-BLOCK-END
