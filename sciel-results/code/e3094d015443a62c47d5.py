import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    t0 = time.time()
    deadline = 80.0
    
    n = P['n']
    base_edges = P['edges']
    
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u, v), max(u, v)))
    
    added_edges = []
    
    def bfs_from(s):
        dist = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist
    
    def compute_metrics():
        dist_matrix = [bfs_from(s) for s in range(n)]
        diam = 0
        num_gt4 = 0
        sum_gt4 = 0
        for i in range(n):
            for j in range(i + 1, n):
                d = dist_matrix[i][j]
                if d > diam:
                    diam = d
                if d > 4:
                    num_gt4 += 1
                    sum_gt4 += d
        return diam, num_gt4, sum_gt4
    
    def is_valid_edge(u, v):
        if u % 2 == v % 2:
            return False
        if (min(u, v), max(u, v)) in edge_set:
            return False
        return True
    
    even_vertices = [i for i in range(n) if i % 2 == 0]
    odd_vertices = [i for i in range(n) if i % 2 == 1]
    
    while True:
        if time.time() - t0 > deadline:
            break
        
        diam, num_gt4, sum_gt4 = compute_metrics()
        if diam <= 4:
            break
        
        best_edge = None
        best_score = None
        
        for u in even_vertices:
            for v in odd_vertices:
                if not is_valid_edge(u, v):
                    continue
                
                adj[u].add(v)
                adj[v].add(u)
                edge_set.add((min(u, v), max(u, v)))
                
                d2, n2, s2 = compute_metrics()
                
                adj[u].discard(v)
                adj[v].discard(u)
                edge_set.discard((min(u, v), max(u, v)))
                
                score = (d2, n2, s2)
                if best_score is None or score < best_score:
                    best_score = score
                    best_edge = (u, v)
        
        if best_edge is None:
            break
        
        u, v = best_edge
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u, v), max(u, v)))
        added_edges.append([int(u), int(v)])
    
    return added_edges
