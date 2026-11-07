"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        u, v = e
        base_edges.add((min(u,v), max(u,v)))
    
    # Build adjacency list for base graph
    adj = [[] for _ in range(n)]
    for u, v in base_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Candidate edges: only between even and odd vertices (to maintain triangle-freeness)
    # and not already in base
    candidates = []
    for u in range(n):
        for v in range(u+1, n):
            if (u - v) % 2 == 1:  # different parity
                if (min(u,v), max(u,v)) not in base_edges:
                    candidates.append((u, v))
    
    # Greedy approach: repeatedly add the edge that most reduces the maximum distance
    added_edges = []
    current_adj = [list(a) for a in adj]
    added_set = set()
    
    def compute_diameter(adj_list):
        """Compute diameter using BFS from each vertex."""
        max_dist = 0
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                node = queue.popleft()
                for neighbor in adj_list[node]:
                    if dist[neighbor] == -1:
                        dist[neighbor] = dist[node] + 1
                        queue.append(neighbor)
                        max_dist = max(max_dist, dist[neighbor])
        return max_dist
    
    # Check initial diameter
    initial_diam = compute_diameter(current_adj)
    
    deadline = time.time() + 80  # Safety margin
    
    # Greedy: keep adding edges until diameter <= 4
    while compute_diameter(current_adj) > 4:
        if time.time() > deadline:
            break
        
        best_edge = None
        best_new_diam = float('inf')
        
        for u, v in candidates:
            if (min(u,v), max(u,v)) in added_set:
                continue
            
            # Try adding this edge
            temp_adj = [list(a) for a in current_adj]
            temp_adj[u].append(v)
            temp_adj[v].append(u)
            
            new_diam = compute_diameter(temp_adj)
            if new_diam < best_new_diam:
                best_new_diam = new_diam
                best_edge = (u, v)
        
        if best_edge is None:
            break
        
        u, v = best_edge
        current_adj[u].append(v)
        current_adj[v].append(u)
        added_edges.append([u, v])
        added_set.add((min(u,v), max(u,v)))
    
    # Verify triangle-freeness
    def is_triangle_free(adj_list):
        for i in range(n):
            for j in adj_list[i]:
                if j > i:
                    for k in adj_list[j]:
                        if k > j and k in adj_list[i]:
                            return False
        return True
    
    if not is_triangle_free(current_adj):
        # Fallback: return empty if something went wrong
        return []
    
    return added_edges
# EVOLVE-BLOCK-END
