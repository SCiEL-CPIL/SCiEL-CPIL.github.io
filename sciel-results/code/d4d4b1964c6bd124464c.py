import time
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adj(n, edge_list):
    adj = [set() for _ in range(n)]
    for u, v in edge_list:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _is_triangle_free(n, adj):
    for u in range(n):
        for v in adj[u]:
            if v > u:
                if adj[u] & adj[v]:
                    return False
    return True

def _diameter(n, adj):
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
        for d in dist:
            if d > max_dist:
                max_dist = d
    return max_dist

def _validate(n, edge_list):
    adj = _build_adj(n, edge_list)
    if not _is_triangle_free(n, adj):
        return False
    if _diameter(n, adj) > 4:
        return False
    return True

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        u, v = e
        if u > v:
            u, v = v, u
        base_edges.add((u, v))
    
    deadline = time.time() + 80
    
    current_edges = list(base_edges)
    current_adj = _build_adj(n, current_edges)
    added = []
    
    all_non_base = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in base_edges:
                all_non_base.append((u, v))
    
    while True:
        if time.time() > deadline - 5:
            break
        diam = _diameter(n, current_adj)
        if diam <= 4:
            break
        
        best_edge = None
        best_diam = diam + 1
        
        for u, v in all_non_base:
            if v in current_adj[u]:
                continue
            if current_adj[u] & current_adj[v]:
                continue
            current_adj[u].add(v)
            current_adj[v].add(u)
            new_diam = _diameter(n, current_adj)
            current_adj[u].discard(v)
            current_adj[v].discard(u)
            
            if new_diam < best_diam:
                best_diam = new_diam
                best_edge = (u, v)
                if best_diam <= 4:
                    break
        
        if best_edge is None:
            break
        
        u, v = best_edge
        current_edges.append((u, v))
        current_adj[u].add(v)
        current_adj[v].add(u)
        added.append([u, v])
    
    final_edges = list(base_edges) + [(u, v) for u, v in added]
    if _validate(n, final_edges):
        return added
    
    return []
# EVOLVE-BLOCK-END
