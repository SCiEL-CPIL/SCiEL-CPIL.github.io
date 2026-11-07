"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _bipartite_check(adj, n):
    """Check if graph is bipartite (triangle-free for this context)."""
    color = [-1] * n
    for i in range(n):
        if color[i] == -1:
            color[i] = 0
            q = deque([i])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if color[v] == -1:
                        color[v] = 1 - color[u]
                        q.append(v)
                    elif color[v] == color[u]:
                        return False
    return True

def _diameter(adj, n):
    """Compute diameter of unweighted graph."""
    max_d = 0
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        max_d = max(max_d, max(dist))
    return max_d

def _build_adj(edges, n):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    n = P['n']
    base_edges = set()
    for u, v in P['edges']:
        base_edges.add((u, v))
        base_edges.add((v, u))
    
    # Initial solution: hub-based approach
    # Use vertices 11 and 12 as hubs
    # 11 (odd) connects to even vertices, 12 (even) connects to odd vertices
    # But we need to be selective to minimize edges while keeping diameter <= 4
    
    # Strategy: Add edges to create shortcuts
    # For diameter 4 on a path of 24 nodes, we need significant shortcuts
    
    # Let's try: connect every k-th node to create a "ladder" structure
    # Optimal for path: add edges (i, i+k) for appropriate k
    
    # Try different configurations
    best_edges = None
    best_count = float('inf')
    
    # Candidate edge sets to try
    candidate_sets = []
    
    # Configuration 1: Two hubs approach
    # Hub at 6 (even) connects to odd vertices at distance > 2
    # Hub at 17 (odd) connects to even vertices at distance > 2
    for hub1, hub2 in [(6, 17), (7, 16), (8, 15), (5, 18), (11, 12), (10, 13), (9, 14)]:
        edges = []
        for v in range(n):
            if v == hub1 or v == hub2:
                continue
            if (hub1 - v) % 2 != 0:  # hub1 connects to opposite parity
                if abs(hub1 - v) > 2:
                    edges.append((min(hub1, v), max(hub1, v)))
            if (hub2 - v) % 2 != 0:  # hub2 connects to opposite parity
                if abs(hub2 - v) > 2:
                    edges.append((min(hub2, v), max(hub2, v)))
        candidate_sets.append(set(edges))
    
    # Configuration 2: Regular pattern - add edges (i, i+8) for all valid i
    for step in [6, 7, 8, 9, 10, 11]:
        edges = []
        for i in range(n):
            j = i + step
            if j < n and (j - i) % 2 == 1:  # Must be opposite parity for bipartite
                edges.append((i, j))
        candidate_sets.append(set(edges))
    
    # Configuration 3: Mixed - some long edges + some medium
    for long_step, med_step in [(12, 6), (10, 5), (14, 7), (8, 4)]:
        edges = []
        for i in range(n):
            j = i + long_step
            if j < n and (j - i) % 2 == 1:
                edges.append((i, j))
        for i in range(n):
            j = i + med_step
            if j < n and (j - i) % 2 == 1:
                edges.append((i, j))
        candidate_sets.append(set(edges))
    
    # Evaluate each candidate
    for edge_set in candidate_sets:
        all_edges = list(base_edges) + [(min(u,v), max(u,v)) for u, v in edge_set if (u,v) not in base_edges and (v,u) not in base_edges]
        # Remove duplicates and self-loops
        unique_edges = set()
        for e in all_edges:
            u, v = e
            if u != v:
                unique_edges.add((min(u, v), max(u, v)))
        
        adj = _build_adj(list(unique_edges), n)
        
        if not _bipartite_check(adj, n):
            continue
        
        diam = _diameter(adj, n)
        if diam <= 4:
            count = len(edge_set)
            if count < best_count:
                best_count = count
                best_edges = edge_set
    
    # If no valid solution found, use the bipartite complete approach
    if best_edges is None:
        best_edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if (v - u) % 2 == 1 and (u, v) not in base_edges and (v, u) not in base_edges:
                    best_edges.add((u, v))
        best_count = len(best_edges)
    
    # Try to improve by removing edges
    if best_edges and time.time() < deadline - 5:
        current_edges = set(best_edges)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for e in list(current_edges):
                trial = current_edges - {e}
                all_edges = list(base_edges) + [(min(u,v), max(u,v)) for u, v in trial]
                unique_edges = set()
                for ed in all_edges:
                    u, v = ed
                    if u != v:
                        unique_edges.add((min(u, v), max(u, v)))
                
                adj = _build_adj(list(unique_edges), n)
                
                if not _bipartite_check(adj, n):
                    continue
                
                diam = _diameter(adj, n)
                if diam <= 4:
                    current_edges = trial
                    best_count = len(trial)
                    improved = True
                    break
            best_edges = current_edges
    
    # Convert to required format
    result = [[u, v] for u, v in sorted(best_edges)]
    # Ensure types are Python int
    result = [[int(u), int(v)] for u, v in result]
    
    return result
# EVOLVE-BLOCK-END
