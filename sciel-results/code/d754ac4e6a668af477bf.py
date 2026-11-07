"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    deadline = time.time() + 50  # Safety margin
    
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        u, v = e
        base_edges.add((min(u,v), max(u,v)))
    
    # Adjacency list for the base graph
    adj = [set() for _ in range(n)]
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    def is_triangle_free(adj, new_edge):
        """Check if adding new_edge to adj keeps graph triangle-free."""
        u, v = new_edge
        # A triangle would be formed if there exists w such that w is neighbor of both u and v
        for w in adj[u]:
            if w != v and w in adj[v]:
                return False
        return True
    
    def compute_diameter(adj):
        """Compute the diameter of the graph using BFS from each vertex."""
        max_dist = 0
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                node = queue.popleft()
                for neighbor in adj[node]:
                    if dist[neighbor] == -1:
                        dist[neighbor] = dist[node] + 1
                        queue.append(neighbor)
            for d in dist:
                if d > max_dist:
                    max_dist = d
        return max_dist
    
    # Get current diameter
    current_diam = compute_diameter(adj)
    
    # Generate candidate edges (not in base, not creating immediate triangle)
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in base_edges:
                candidates.append((u, v))
    
    # Greedy approach: repeatedly add the edge that most reduces diameter
    added_edges = []
    added_set = set()
    
    while compute_diameter(adj) > 4:
        if time.time() > deadline:
            break
        
        best_edge = None
        best_diam = current_diam + 1
        best_reduction = 0
        
        for u, v in candidates:
            if (u, v) in added_set:
                continue
            if not is_triangle_free(adj, (u, v)):
                continue
            
            # Try adding this edge
            adj[u].add(v)
            adj[v].add(u)
            
            new_diam = compute_diameter(adj)
            
            # Remove the edge
            adj[u].remove(v)
            adj[v].remove(u)
            
            reduction = current_diam - new_diam
            if reduction > best_reduction or (reduction == best_reduction and new_diam < best_diam):
                best_reduction = reduction
                best_diam = new_diam
                best_edge = (u, v)
        
        if best_edge is None:
            # Can't reduce further with greedy, break
            break
        
        # Add the best edge
        u, v = best_edge
        adj[u].add(v)
        adj[v].add(u)
        added_set.add((u, v))
        added_edges.append([u, v])
        current_diam = best_diam
    
    # Return as list of [u, v] with Python ints
    return [[int(u), int(v)] for u, v in added_edges]
# EVOLVE-BLOCK-END
