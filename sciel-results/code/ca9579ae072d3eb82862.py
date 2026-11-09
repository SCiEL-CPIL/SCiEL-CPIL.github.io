import time
from collections import deque

def _bfs_diameter(adj):
    n = len(adj)
    diam = 0
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        for d in dist:
            if d == -1:
                return float('inf')
            if d > diam:
                diam = d
    return diam

def construct():
    t0 = time.time()
    n = 16
    half = n // 2
    edges = []
    for u in range(half):
        for v in range(half, n):
            edges.append([u, v])
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    diam = _bfs_diameter(adj)
    if diam != 2:
        return []
    if time.time() - t0 > 55:
        return [[int(u), int(v)] for u, v in edges]
    return [[int(u), int(v)] for u, v in edges]
