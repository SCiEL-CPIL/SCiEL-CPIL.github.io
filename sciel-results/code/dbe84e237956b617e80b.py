"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from collections import deque
import random

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add(tuple(sorted(e)))
    
    adj = [[] for _ in range(n)]
    for u, v in base_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    def bfs_all(adj):
        dist = [[-1]*n for _ in range(n)]
        for start in range(n):
            dist[start][start] = 0
            q = deque([start])
            while q:
                node = q.popleft()
                for nb in adj[node]:
                    if dist[start][nb] == -1:
                        dist[start][nb] = dist[start][node] + 1
                        q.append(nb)
        return dist
    
    def get_diameter(dist):
        max_d = 0
        for i in range(n):
            for j in range(n):
                if dist[i][j] > max_d:
                    max_d = dist[i][j]
        return max_d
    
    def check_triangle_free(adj, added_edges):
        adj_set = [set(nb) for nb in adj]
        for u, v in added_edges:
            adj_set[u].add(v)
            adj_set[v].add(u)
        for i in range(n):
            for j in range(i+1, n):
                if j in adj_set[i]:
                    for k in range(j+1, n):
                        if k in adj_set[i] and k in adj_set[j]:
                            return False
        return True
    
    candidates = []
    for u in range(n):
        for v in range(u+1, n):
            if (u, v) not in base_edges:
                candidates.append((u, v))
    
    # Greedy approach
    added = []
    cur_adj = [list(nb) for nb in adj]
    
    for iteration in range(10):
        dist = bfs_all(cur_adj)
        diam = get_diameter(dist)
        if diam <= 4:
            break
        
        best_edge = None
        best_diam = diam
        
        for cand in candidates:
            u, v = cand
            if (u, v) in added or (v, u) in added:
                continue
            common = set(cur_adj[u]) & set(cur_adj[v])
            if common:
                continue
            
            cur_adj[u].append(v)
            cur_adj[v].append(u)
            new_dist = bfs_all(cur_adj)
            new_diam = get_diameter(new_dist)
            
            if new_diam < best_diam:
                best_diam = new_diam
                best_edge = cand
            
            cur_adj[u].remove(v)
            cur_adj[v].remove(u)
        
        if best_edge is None:
            break
        
        u, v = best_edge
        cur_adj[u].append(v)
        cur_adj[v].append(u)
        added.append((u, v))
    
    if check_triangle_free(adj, added):
        dist = bfs_all(cur_adj)
        if get_diameter(dist) <= 4:
            return [[int(u), int(v)] for u, v in added]
    
    # Fallback random search
    random.seed(42)
    for k in range(1, 12):
        for _ in range(30000):
            sample = random.sample(candidates, k)
            adj_set = [set(nb) for nb in adj]
            valid = True
            for u, v in sample:
                if adj_set[u] & adj_set[v]:
                    valid = False
                    break
                adj_set[u].add(v)
                adj_set[v].add(u)
            if not valid:
                continue
            test_adj = [list(nb) for nb in adj]
            for u, v in sample:
                test_adj[u].append(v)
                test_adj[v].append(u)
            d = bfs_all(test_adj)
            if get_diameter(d) <= 4:
                return [[int(u), int(v)] for u, v in sample]
    
    return []

# EVOLVE-BLOCK-END
