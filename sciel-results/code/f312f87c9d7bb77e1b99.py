import numpy as np
import time
import random
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order

def _check_girth(n, edges, g):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    half_g = g // 2
    for s in range(n):
        dist = [-1] * n
        pred = [-1] * n
        dist[s] = 0
        queue = [s]
        for u in queue:
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    pred[v] = u
                    if dist[v] < half_g:
                        queue.append(v)
                elif pred[u] != v and pred[v] != u:
                    if dist[u] + dist[v] + 1 < g:
                        return False
    return True

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    N = 1000
    while time.time() < deadline:
        # Generate random 3-regular graph
        # Use a simple method: Random permutation of half-edges
        half_edges = []
        for i in range(N):
            for _ in range(k):
                half_edges.append(i)
        
        random.shuffle(half_edges)
        edges = []
        valid = True
        for i in range(0, len(half_edges), 2):
            u = half_edges[i]
            v = half_edges[i+1]
            if u == v:
                valid = False
                break
            if u > v: u, v = v, u
            if (u, v) in edges:
                valid = False
                break
            edges.append((u, v))
            
        if not valid:
            continue
            
        if _check_girth(N, edges, g):
            return N, edges
            
        N += 100

    return N, edges
