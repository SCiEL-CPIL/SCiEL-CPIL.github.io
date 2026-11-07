"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for u, v in P['edges']:
        base_edges.add((u, v))
        base_edges.add((v, u))
    
    def build_adj(added_edges):
        adj = [[] for _ in range(n)]
        for u, v in P['edges']:
            adj[u].append(v)
            adj[v].append(u)
        for u, v in added_edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj
    
    def is_triangle_free(adj):
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    common = set(adj[i]) & set(adj[j])
                    if common:
                        return False
        return True
    
    def diameter(adj):
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
                        queue.append(v)
            for d in dist:
                if d > max_dist:
                    max_dist = d
        return max_dist
    
    def is_connected(adj):
        visited = [False] * n
        queue = deque([0])
        visited[0] = True
        count = 0
        while queue:
            u = queue.popleft()
            count += 1
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        return count == n
    
    def validate(added_edges):
        added_set = set()
        for u, v in added_edges:
            if (u, v) in base_edges or (v, u) in base_edges:
                return False
            if (u, v) in added_set or (v, u) in added_set:
                return False
            added_set.add((u, v))
        adj = build_adj(added_edges)
        if not is_connected(adj):
            return False
        if not is_triangle_free(adj):
            return False
        if diameter(adj) > 4:
            return False
        return True
    
    # Try specific candidate solutions
    candidates = [
        # 3-edge attempts
        [[0, 8], [8, 16], [16, 23]],
        [[0, 7], [7, 16], [16, 23]],
        [[0, 6], [6, 12], [12, 18], [18, 23]],  # 4 edges
        [[0, 7], [7, 14], [14, 21], [21, 23]],  # 4 edges
        [[0, 5], [5, 10], [10, 15], [15, 20], [20, 23]],  # 5 edges
        [[0, 6], [6, 12], [12, 18], [18, 23]],  # 4 edges
        [[0, 7], [7, 13], [13, 19], [19, 23]],  # 4 edges
        [[0, 8], [8, 15], [15, 22], [22, 23]],  # 4 edges
        [[0, 9], [9, 17], [17, 23]],  # 3 edges
        [[0, 8], [8, 16], [16, 23]],  # 3 edges
        [[0, 7], [7, 15], [15, 23]],  # 3 edges
        [[0, 6], [6, 14], [14, 22], [22, 23]],  # 4 edges
        [[0, 5], [5, 12], [12, 19], [19, 23]],  # 4 edges
        [[0, 6], [6, 13], [13, 20], [20, 23]],  # 4 edges
        [[0, 7], [7, 14], [14, 21], [21, 23]],  # 4 edges
        [[0, 8], [8, 15], [15, 22], [22, 23]],  # 4 edges
        [[0, 9], [9, 16], [16, 23]],  # 3 edges
        [[0, 10], [10, 18], [18, 23]],  # 3 edges
        [[0, 11], [11, 19], [19, 23]],  # 3 edges
        [[0, 12], [12, 20], [20, 23]],  # 3 edges
        [[0, 12], [12, 23]],  # 2 edges
    ]
    
    for cand in candidates:
        if validate(cand):
            return cand
    
    # Fallback: hub strategy
    hub = 12
    added = []
    for v in range(n):
        if v % 2 != 0:
            if (hub, v) not in base_edges and (v, hub) not in base_edges:
                added.append([hub, v])
    return added

# EVOLVE-BLOCK-END
