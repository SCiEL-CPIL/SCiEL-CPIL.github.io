"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 60
    
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((e[0], e[1]))
        base_edges.add((e[1], e[0]))
    
    # Build adjacency as sets
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if (u, v) in base_edges and u != v:
                adj[u].add(v)
    
    def is_triangle_free(u, v):
        # Adding edge (u,v) creates a triangle if there's a common neighbor
        return len(adj[u] & adj[v]) == 0
    
    def compute_diameter():
        max_dist = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
            m = max(dist)
            if m > max_dist:
                max_dist = m
        return max_dist
    
    # Generate all candidate edges (not in base, not self-loops)
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in base_edges and (v, u) not in base_edges:
                candidates.append((u, v))
    
    current_diam = compute_diameter()
    added = []
    
    # Greedy: repeatedly add the edge that most reduces diameter
    while current_diam > 4:
        if time.time() > deadline:
            break
        
        best_edge = None
        best_diam = current_diam
        best_score = -1  # We want to maximize reduction
        
        for edge in candidates:
            if time.time() > deadline:
                break
            u, v = edge
            if not is_triangle_free(u, v):
                continue
            
            # Tentatively add edge
            adj[u].add(v)
            adj[v].add(u)
            d = compute_diameter()
            adj[u].remove(v)
            adj[v].remove(u)
            
            # Score: prefer edges that reduce diameter most
            reduction = current_diam - d
            if reduction > best_score:
                best_score = reduction
                best_diam = d
                best_edge = edge
        
        if best_edge is None:
            break
        
        u, v = best_edge
        adj[u].add(v)
        adj[v].add(u)
        added.append([u, v])
        candidates.remove(best_edge)
        current_diam = best_diam
    
    # Verify the result
    if current_diam > 4:
        # Try a different approach: more aggressive search
        # Reset and try adding edges that maximize connectivity
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(n):
                if (u, v) in base_edges and u != v:
                    adj[u].add(v)
        
        added = []
        current_diam = compute_diameter()
        
        # Try all pairs of non-adjacent vertices and add if it helps
        while current_diam > 4 and time.time() < deadline:
            found = False
            for edge in list(candidates):
                if time.time() > deadline:
                    break
                u, v = edge
                if not is_triangle_free(u, v):
                    continue
                adj[u].add(v)
                adj[v].add(u)
                d = compute_diameter()
                if d <= current_diam:
                    added.append([u, v])
                    candidates.remove(edge)
                    current_diam = d
                    found = True
                    break
                else:
                    adj[u].remove(v)
                    adj[v].remove(u)
            if not found:
                break
    
    # Ensure return type is correct
    result = []
    for edge in added:
        if isinstance(edge, tuple):
            result.append([int(edge[0]), int(edge[1])])
        else:
            result.append([int(edge[0]), int(edge[1])])
    return result
# EVOLVE-BLOCK-END
