"""
Optimized solution for the dense graph task.
Constructs K_{8,8}, which is proven to be the maximum edge count
for a triangle-free graph on 16 vertices (Mantel's Theorem).
Since any triangle in a diameter-2 graph typically renders its edges
non-critical (distance drops to 2 instead of increasing >2),
K_{8,8} is the optimal solution.
"""

import time
from collections import deque

P = {'n': 16}
START_TIME = time.time()
TIME_LIMIT = 85

def bfs_distances(adj, start):
    n = len(adj)
    dist = [-1] * n
    dist[start] = 0
    q = deque([start])
    while q:
        u = q.popleft()
        du = dist[u]
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = du + 1
                q.append(v)
    return dist

def compute_diameter(adj):
    n = len(adj)
    max_dist = 0
    for i in range(n):
        dists = bfs_distances(adj, i)
        if -1 in dists:
            return -1
        max_dist = max(max_dist, max(dists))
    return max_dist

def is_diameter_two(adj):
    return compute_diameter(adj) == 2

def edge_is_critical(adj, u, v):
    # Remove edge
    try:
        adj[u].remove(v)
    except ValueError:
        pass
    try:
        adj[v].remove(u)
    except ValueError:
        pass
    
    new_diam = compute_diameter(adj)
    
    # Re-add edge
    adj[u].append(v)
    adj[v].append(u)
    
    return new_diam > 2 or new_diam == -1

def construct():
    n = P['n']
    
    # Construct K_{8,8}
    # Partitions: 0-7 and 8-15
    partition_a = list(range(0, 8))
    partition_b = list(range(8, 16))
    
    edges = []
    for u in partition_a:
        for v in partition_b:
            edges.append([u, v])
            
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Verify diameter is 2
    if not is_diameter_two(adj):
        # Fallback: Star graph (diameter 2, all edges critical as bridges)
        # This case should never happen for K_{8,8}
        return [[0, v] for v in range(1, n)]
        
    # Verify all edges are critical
    # This is a sanity check. Mathematically, K_{8,8} edges are critical.
    # For n=16, this loop is fast.
    for u, v in edges:
        if not edge_is_critical(adj, u, v):
            # If any edge is not critical, we have a problem.
            # However, K_{8,8} is known to be valid.
            # If this fails, we might try to repair or fall back, 
            # but there is no better fallback with >15 edges.
            # We will just return the star as a safe, valid (though low score) graph.
            return [[0, v] for v in range(1, n)]

    return edges
