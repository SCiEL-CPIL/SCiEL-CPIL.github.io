import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # The maximum n is 35.
    # We construct the (35, 16, 6, 8) Strongly Regular Graph.
    # Vertices are the 35 triples from {0, 1, 2, 3, 4, 5, 6}.
    # Two vertices (triples) are adjacent if their intersection size is NOT 1.
    # i.e., intersection size is 0 or 2.
    
    vertices = list(combinations(range(7), 3))
    n = len(vertices)
    
    # Map tuple to index
    idx_map = {v: i for i, v in enumerate(vertices)}
    
    adj = np.zeros((n, n), dtype=bool)
    
    edges = []
    
    for i in range(n):
        for j in range(i + 1, n):
            u = vertices[i]
            v = vertices[j]
            # Calculate intersection size
            inter_size = len(set(u).intersection(v))
            
            # Adjacent if inter_size != 1
            if inter_size != 1:
                adj[i, j] = True
                adj[j, i] = True
                edges.append([i, j])
                
    # Verification function
    def is_valid(n, edges):
        adj = np.zeros((n, n), dtype=bool)
        for u, v in edges:
            adj[u, v] = True
            adj[v, u] = True
            
        # Check K4
        for i in range(n):
            for j in range(i + 1, n):
                if not adj[i, j]: continue
                for k in range(j + 1, n):
                    if not adj[i, k] or not adj[j, k]: continue
                    for l in range(k + 1, n):
                        if adj[i, l] and adj[j, l] and adj[k, l]:
                            return False
                            
        # Check Independent Set of 6
        comp = ~adj
        np.fill_diagonal(comp, False)
        
        for i in range(n):
            for j in range(i + 1, n):
                if not comp[i, j]: continue
                for k in range(j + 1, n):
                    if not comp[i, k] or not comp[j, k]: continue
                    for l in range(k + 1, n):
                        if not comp[i, l] or not comp[j, l] or not comp[k, l]: continue
                        for m in range(l + 1, n):
                            if comp[i, m] and comp[j, m] and comp[k, m] and comp[l, m]:
                                for p in range(m + 1, n):
                                    if comp[i, p] and comp[j, p] and comp[k, p] and comp[l, p] and comp[m, p]:
                                        return False
                                        
        return True

    # Verify the constructed graph
    if is_valid(n, edges):
        return {'n': n, 'edges': edges}
        
    # Fallback: Local search if verification fails (should not happen)
    # But for safety, return a smaller known valid graph if this fails.
    # The Paley graph of order 17 is valid.
    squares = {x * x % 17 for x in range(1, 17)}
    edges_17 = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
    if is_valid(17, edges_17):
        return {'n': 17, 'edges': edges_17}
        
    return {'n': 1, 'edges': []}
