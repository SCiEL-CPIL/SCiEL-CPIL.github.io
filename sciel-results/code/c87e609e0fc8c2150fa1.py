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
    
    # Initialize distance matrix
    dist = [[0 if i == j else abs(i - j) for j in range(n)] for i in range(n)]
    
    adj = [set() for _ in range(n)]
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
        
    added_edges = []
    
    def creates_triangle(u, v):
        return len(adj[u] & adj[v]) > 0

    def get_diameter():
        max_d = 0
        for i in range(n):
            for j in range(n):
                if dist[i][j] > max_d:
                    max_d = dist[i][j]
        return max_d

    def add_edge(u, v):
        adj[u].add(v)
        adj[v].add(u)
        added_edges.append((u, v))
        # Update distances
        for i in range(n):
            for j in range(n):
                new_d = min(dist[i][j], dist[i][u] + 1 + dist[v][j], dist[i][v] + 1 + dist[u][j])
                if new_d < dist[i][j]:
                    dist[i][j] = new_d
                    dist[j][i] = new_d

    def remove_edge(u, v):
        adj[u].remove(v)
        adj[v].remove(u)
        added_edges.remove((u, v))
        # Recalculate distances from scratch (simpler and safe)
        # Or just re-run Floyd Warshall?
        # For simplicity in removal, just recompute BFS or Floyd.
        # Since n=24, recompute is fast.
        new_dist = [[0 if i == j else abs(i - j) for j in range(n)] for i in range(n)]
        # Apply base edges
        # Actually, just run Floyd Warshall on the current adj
        # Initialize with inf
        INF = 100
        new_dist = [[INF] * n for _ in range(n)]
        for i in range(n):
            new_dist[i][i] = 0
        for u, v in base_edges:
            new_dist[u][v] = 1
            new_dist[v][u] = 1
        for u, v in added_edges:
            new_dist[u][v] = 1
            new_dist[v][u] = 1
        # Floyd Warshall
        for k in range(n):
            for i in range(n):
                if new_dist[i][k] == INF: continue
                for j in range(n):
                    if new_dist[k][j] == INF: continue
                    if new_dist[i][j] > new_dist[i][k] + new_dist[k][j]:
                        new_dist[i][j] = new_dist[i][k] + new_dist[k][j]
        # Copy back
        for i in range(n):
            for j in range(n):
                dist[i][j] = new_dist[i][j]

    # Generate candidates
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in base_edges:
                candidates.append((u, v))
                
    # Sort by distance descending (longer jumps first)
    candidates.sort(key=lambda x: x[1] - x[0], reverse=True)

    # Greedy Phase
    while get_diameter() > 4:
        best_edge = None
        best_diam = float('inf')
        
        for u, v in candidates:
            if (u, v) in added_edges:
                continue
            if creates_triangle(u, v):
                continue
            
            # Simulate add
            # To avoid modifying and reverting complex state, 
            # we can just check if this edge helps.
            # For speed, we just pick the one with max distance reduction heuristic?
            # Let's do a quick check: does it reduce the current max distance?
            # Compute new diameter approximately?
            # Just try adding, checking, and removing.
            
            # Save state
            old_dist = copy.deepcopy(dist)
            
            add_edge(u, v)
            new_diam = get_diameter()
            
            # Remove
            remove_edge(u, v)
            # Restore dist (remove_edge recalcs, but we saved old_dist)
            # Actually remove_edge recalcs correctly based on added_edges list.
            # But add_edge added to list. remove_edge removed from list.
            # So dist is correct after remove_edge.
            
            if new_diam < best_diam:
                best_diam = new_diam
                best_edge = (u, v)
                
        if best_edge is None:
            break
            
        add_edge(best_edge[0], best_edge[1])
        
    # Local Search: Try to remove edges
    changed = True
    while changed:
        changed = False
        for i in range(len(added_edges) - 1, -1, -1):
            u, v = added_edges[i]
            # Try removing
            remove_edge(u, v)
            if get_diameter() <= 4:
                changed = True
            else:
                # Restore
                add_edge(u, v)
                
    return [[u, v] for u, v in added_edges]

# EVOLVE-BLOCK-END
