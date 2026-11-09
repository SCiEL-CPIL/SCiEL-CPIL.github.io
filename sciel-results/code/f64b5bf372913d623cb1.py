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
    t0 = time.time()
    n = 16
    half = n // 2
    # K_{8,8}: complete bipartite graph on 16 vertices
    # - diameter exactly 2 (any two vertices in the same part share all 8 neighbors in the other part)
    # - every edge is essential: removing edge (u,v) with u in part A, v in part B
    #   leaves no path of length 1 or 2 between u and v (no common neighbor),
    #   so dist(u,v) becomes 3, increasing diameter above 2
    # - 64 edges is the maximum possible for a triangle-free graph (Mantel's theorem),
    #   and any graph with diameter 2 and all edges essential on 16 vertices
    #   cannot exceed this bound without creating a triangle or losing edge-essentiality
    edge_list = [[int(u), int(v)] for u in range(half) for v in range(half, n)]

    # Verify diameter is exactly 2
    adj = [[] for _ in range(n)]
    for u, v in edge_list:
        adj[u].append(v)
        adj[v].append(u)
    if _bfs_diameter(adj) != 2:
        return []

    # Deadline guard
    if time.time() - t0 > 55:
        return edge_list

    return edge_list
