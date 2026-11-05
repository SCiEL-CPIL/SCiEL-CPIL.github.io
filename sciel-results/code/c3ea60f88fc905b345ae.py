"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque


def priority(adjacency):
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # Precompute neighbor lists for faster traversal
    neighbors = [np.where(adj[i])[0] for i in range(n)]
    degrees = np.array([len(neigh) for neigh in neighbors], dtype=float)
    
    # Compute Betweenness Centrality (Brandes' Algorithm)
    # This captures global structural importance (bridges vs leaves)
    bc = np.zeros(n)
    
    for s in range(n):
        prev = [[] for _ in range(n)]
        sigma = np.zeros(n)
        sigma[s] = 1.0
        dist = np.full(n, -1)
        dist[s] = 0
        
        q = deque([s])
        while q:
            v = q.popleft()
            for w in neighbors[v]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    q.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    prev[w].append(v)
        
        delta = np.zeros(n)
        # Process vertices in reverse order of distance from source
        # We only care about reachable vertices
        valid_nodes = [i for i in range(n) if dist[i] >= 0]
        valid_nodes.sort(key=lambda x: dist[x], reverse=True)
        
        for w in valid_nodes:
            for v in prev[w]:
                if dist[v] == dist[w] - 1:
                    if sigma[w] > 0:
                        delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != s:
                bc[w] += delta[w]
                
    # Normalize betweenness centrality to [0, 1] to combine with degree
    if bc.max() > 0:
        bc_norm = bc / bc.max()
    else:
        bc_norm = bc
        
    # Heuristic:
    # 1. Prefer vertices with LOW degree (standard MIS greedy).
    # 2. Boost vertices with HIGH betweenness centrality.
    #    Why? Vertices with high betweenness are often "bridges" or "cut-points".
    #    Picking a bridge early separates the graph into components,
    #    allowing the greedy algorithm to independently maximize the
    #    independent set in each component without cross-interference.
    #    This is particularly effective in graphs with community structures
    #    like Erdos-Renyi graphs which may have sparse bottlenecks.
    
    score = -degrees + bc_norm
    
    return score
# EVOLVE-BLOCK-END
