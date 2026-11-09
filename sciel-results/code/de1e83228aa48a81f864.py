import time
from collections import deque

def _bfs_diameter(adj):
    """Return the diameter of the graph given by adjacency lists.
    Returns float('inf') if the graph is disconnected."""
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
    start_time = time.time()
    n = 16
    half = n // 2
    edges = []
    
    # Construct K_{8,8} - the complete bipartite graph
    # By Mantel's theorem, this is the maximum number of edges in a
    # triangle-free graph on 16 vertices (64 edges).
    # Every edge is essential: for edge (u,v) with u in {0..7}, v in {8..15},
    # there is no common neighbor, so removing (u,v) makes dist(u,v)=3.
    for u in range(half):
        for v in range(half, n):
            edges.append([u, v])
    
    # Build adjacency list and verify diameter is exactly 2
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    diam = _bfs_diameter(adj)
    if diam != 2:
        return []
    
    # Safety deadline check
    if time.time() - start_time > 55:
        return [[int(u), int(v)] for u, v in edges]
    
    # Return edges as list of [u, v] with Python ints
    return [[int(u), int(v)] for u, v in edges]
