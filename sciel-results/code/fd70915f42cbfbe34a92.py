"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    deadline = time.time() + 80
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((min(e[0], e[1]), max(e[0], e[1])))
    
    # Precompute adjacency for base
    base_adj = [set() for _ in range(n)]
    for u, v in base_edges:
        base_adj[u].add(v)
        base_adj[v].add(u)
    
    # Candidate edges: gap >= 3
    cands = [(u, v) for u in range(n) for v in range(u + 3, n)]
    cands.sort(key=lambda x: x[1] - x[0], reverse=True)
    
    def get_adj(added):
        adj = [set(base_adj[i]) for i in range(n)]
        for u, v in added:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def is_tri_free(adj):
        for u in range(n):
            for v in adj[u]:
                for w in adj[v]:
                    if w > v and w in adj[u]:
                        return False
        return True
    
    def diam(adj):
        md = 0
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                c = q.popleft()
                for nb in adj[c]:
                    if dist[nb] < 0:
                        dist[nb] = dist[c] + 1
                        if dist[nb] > md:
                            md = dist[nb]
                        q.append(nb)
        return md
    
    # Greedy: add most useful edge first
    added = []
    adj = [set(base_adj[i]) for i in range(n)]
    
    while time.time() < deadline - 2:
        d = diam(adj)
        if d <= 4:
            break
        best_e = None
        best_d = d
        for e in cands:
            if e in added:
                continue
            u, v = e
            # Quick triangle check: u and v must not share a neighbor
            common = adj[u] & adj[v]
            if common:
                continue
            # Add edge temporarily
            adj[u].add(v)
            adj[v].add(u)
            nd = diam(adj)
            adj[u].discard(v)
            adj[v].discard(u)
            if nd < best_d:
                best_d = nd
                best_e = e
        if best_e is None:
            break
        u, v = best_e
        adj[u].add(v)
        adj[v].add(u)
        added.append(best_e)
    
    if diam(adj) <= 4:
        return [[u, v] for u, v in added]
    
    # Fallback: dense set
    res = []
    for u in range(n):
        for v in range(u + 3, n):
            if (u, v) not in base_edges:
                res.append([u, v])
    return res
# EVOLVE-BLOCK-END
