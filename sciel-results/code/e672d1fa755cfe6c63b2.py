# EVOLVE-BLOCK-START
from itertools import combinations
from collections import deque
import copy

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((min(e[0], e[1]), max(e[0], e[1])))
    
    # Initialize adjacency for base graph
    adj = [set() for _ in range(n)]
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Precompute valid edges that do not form a triangle with base path
    # A single edge (u,v) forms a triangle with path if there is w such that (u,w) and (v,w) are in base.
    # In a path, this happens if dist(u,v) == 2.
    # So valid edges must have dist(u,v) != 2 and dist(u,v) != 1.
    valid_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in base_edges and v - u != 2:
                valid_edges.append((u, v))
                
    # Sort valid edges by distance descending? Or just keep them.
    # For backtracking, order matters for pruning efficiency.
    # Longer edges reduce diameter more, so try them first?
    valid_edges.sort(key=lambda x: x[1] - x[0], reverse=True)
    
    # State for backtracking
    current_edges = []
    current_adj = [set(s) for s in adj]
    
    def is_triangle_free(u, v):
        # Check if adding (u,v) creates a triangle
        # i.e., if any neighbor of u is also a neighbor of v
        return len(current_adj[u] & current_adj[v]) == 0
        
    def add_edge(u, v):
        current_adj[u].add(v)
        current_adj[v].add(u)
        current_edges.append((u, v))
        
    def remove_edge(u, v):
        current_adj[u].remove(v)
        current_adj[v].remove(u)
        current_edges.pop()
        
    def get_diameter():
        # Floyd Warshall on n=24 is fast
        INF = 100
        dist = [[INF] * n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0
        for u, v in base_edges:
            dist[u][v] = 1
            dist[v][u] = 1
        for u, v in current_edges:
            dist[u][v] = 1
            dist[v][u] = 1
            
        for k in range(n):
            for i in range(n):
                if dist[i][k] == INF: continue
                for j in range(n):
                    if dist[k][j] == INF: continue
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        
        max_d = 0
        for i in range(n):
            for j in range(n):
                if dist[i][j] > max_d:
                    max_d = dist[i][j]
        return max_d
        
    # Backtracking search for minimum k
    # We try k=1, 2, 3...
    # Since we suspect 3 is optimal, we start from 1.
    
    def backtrack(k, start_idx):
        if k == 0:
            return get_diameter() <= 4
            
        for i in range(start_idx, len(valid_edges)):
            u, v = valid_edges[i]
            if is_triangle_free(u, v):
                add_edge(u, v)
                if backtrack(k - 1, i + 1):
                    return True
                remove_edge(u, v)
        return False
        
    # Try increasing k
    for k in range(1, 10):
        if backtrack(k, 0):
            # Convert to list of lists of ints
            return [[int(u), int(v)] for u, v in current_edges]
            
    # Fallback (should not happen)
    return []

# EVOLVE-BLOCK-END
