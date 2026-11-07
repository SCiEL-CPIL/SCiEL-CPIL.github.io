"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    deadline = time.time() + 80  # safety margin before 90s limit
    
    # Build base adjacency
    adj = [set() for _ in range(n)]
    base_edges = set()
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        base_edges.add(frozenset({u, v}))
    
    def bfs_diameter():
        """Compute diameter of current graph using BFS from each node."""
        max_diam = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            queue = [src]
            for i in range(len(queue)):
                u = queue[i]
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            # Check connectivity
            for d in dist:
                if d == -1:
                    return -1  # not connected
            max_diam = max(max_diam, max(dist))
        return max_diam
    
    def is_triangle_free_with_edge(u, v):
        """Check if adding edge (u,v) keeps graph triangle-free."""
        # For each common neighbor, would create a triangle
        common = adj[u] & adj[v]
        if common:
            return False
        return True
    
    def add_edge(u, v):
        adj[u].add(v)
        adj[v].add(u)
    
    def remove_edge(u, v):
        adj[u].discard(v)
        adj[v].discard(u)
    
    # Start with the base graph (just the path)
    added_edges = []
    
    # First, check if the path already has diameter <= 4
    current_diam = bfs_diameter()
    if current_diam <= 4:
        return []
    
    # Greedy: repeatedly add the edge that most reduces diameter
    # but we must maintain triangle-free property
    
    # Pre-compute all valid candidate edges (triangle-free with current graph)
    # and sort them by some heuristic (e.g., by how much they shorten paths)
    
    while True:
        if time.time() > deadline:
            break
        
        current_diam = bfs_diameter()
        if current_diam <= 4:
            break
        if current_diam == -1:
            # Should not happen since path is connected
            break
        
        # Find all candidate edges that can be added while keeping triangle-free
        candidates = []
        for u in range(n):
            for v in range(u + 1, n):
                if v in adj[u]:  # already an edge
                    continue
                if frozenset({u, v}) in base_edges:
                    continue
                if not is_triangle_free_with_edge(u, v):
                    continue
                candidates.append((u, v))
        
        if not candidates:
            break
        
        # Try each candidate and pick the one that gives the smallest diameter
        best_edge = None
        best_diam = current_diam
        
        for u, v in candidates:
            # Try adding this edge
            add_edge(u, v)
            new_diam = bfs_diameter()
            remove_edge(u, v)
            
            if new_diam != -1 and new_diam < best_diam:
                best_diam = new_diam
                best_edge = (u, v)
        
        if best_edge is None:
            # No improvement possible
            break
        
        u, v = best_edge
        add_edge(u, v)
        added_edges.append([int(u), int(v)])
    
    return added_edges
# EVOLVE-BLOCK-END
