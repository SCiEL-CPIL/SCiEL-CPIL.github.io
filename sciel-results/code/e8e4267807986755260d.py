"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _bfs_dist(n, adj, start):
    dist = [-1] * n
    dist[start] = 0
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist

def _diameter(n, adj):
    d0 = _bfs_dist(n, adj, 0)
    if any(d == -1 for d in d0):
        return -1
    diam = 0
    for i in range(n):
        d = _bfs_dist(n, adj, i)
        if any(x == -1 for x in d):
            return -1
        diam = max(diam, max(d))
    return diam

def _is_triangle_free(n, adj):
    for i in range(n):
        ni = adj[i]
        for j in ni:
            if j > i:
                for k in ni:
                    if k > j and k in adj[j]:
                        return False
    return True

def construct():
    n = P['n']
    base = [tuple(e) for e in P['edges']]
    base_set = set(base)
    
    deadline = time.time() + 80
    
    # Candidate edges: different parity pairs not in base (bipartite => triangle-free)
    cands = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u % 2) != (v % 2):
                if (u, v) not in base_set:
                    cands.append((u, v))
    
    current = []
    cur_set = set()
    
    def _get_diam():
        all_e = base + list(current)
        adj = _build_adj(n, all_e)
        return _diameter(n, adj)
    
    cur_diam = _get_diam()
    
    while cur_diam > 4:
        if time.time() > deadline:
            break
        best_edge = None
        best_diam = cur_diam
        for c in cands:
            if c in cur_set:
                continue
            all_e = base + list(current) + [c]
            adj = _build_adj(n, all_e)
            d = _diameter(n, adj)
            if d == -1:
                continue
            if d < best_diam:
                best_diam = d
                best_edge = c
                if d <= 4:
                    break
        if best_edge is None:
            break
        current.append(best_edge)
        cur_set.add(best_edge)
        cur_diam = best_diam
    
    all_e = base + list(current)
    adj = _build_adj(n, all_e)
    if not _is_triangle_free(n, adj):
        return []
    d = _diameter(n, adj)
    if d > 4 or d == -1:
        return []
    
    return [[int(u), int(v)] for u, v in current]
# EVOLVE-BLOCK-END
