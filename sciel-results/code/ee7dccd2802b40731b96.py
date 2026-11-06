import time
import random
import numpy as np
from itertools import combinations

DEADLINE = 145.0
START_TIME = time.time()

def _time_left():
    return DEADLINE - (time.time() - START_TIME)

n = 16
# Precompute all 8-vertex subsets
all_subsets_8 = list(combinations(range(n), 8))
num_subsets = len(all_subsets_8)

# For each subset, precompute the vertex indices
subset_indices = np.array(all_subsets_8)

def count_q3(adj):
    """Count Q3 subgraphs using vectorized numpy operations."""
    count = 0
    for i in range(num_subsets):
        verts = subset_indices[i]
        sub = adj[np.ix_(verts, verts)]
        degs = sub.sum(axis=1)
        if not np.all(degs == 3):
            continue
        # Check bipartite
        visited = np.zeros(8, dtype=bool)
        color = np.zeros(8, dtype=int)
        bipartite = True
        for start in range(8):
            if visited[start]:
                continue
            visited[start] = True
            color[start] = 0
            queue = [start]
            while queue:
                u = queue.pop(0)
                for v in range(8):
                    if sub[u, v]:
                        if not visited[v]:
                            visited[v] = True
                            color[v] = 1 - color[u]
                            queue.append(v)
                        elif color[v] == color[u]:
                            bipartite = False
                            break
                if not bipartite:
                    break
            if not bipartite:
                break
        if bipartite:
            count += 1
    return count

def construct():
    global START_TIME
    START_TIME = time.time()
    random.seed(42)
    
    # Start from K7+K7+K2 seed (57 edges)
    adj = np.zeros((16,16), dtype=bool)
    for i in range(7):
        for j in range(i+1,7):
            adj[i,j] = adj[j,i] = True
    for i in range(7,14):
        for j in range(i+1,14):
            adj[i,j] = adj[j,i] = True
    adj[14,15] = adj[15,14] = True
    
    initial_q3 = count_q3(adj)
    initial_edges = adj.sum() // 2
    print(f"Initial: {initial_edges} edges, {initial_q3} Q3s")
    
    best_adj = adj.copy()
    best_edges = initial_edges
    
    # Local search: try adding edges
    improved = True
    while improved and _time_left() > 5.0:
        improved = False
        for i in range(16):
            for j in range(i+1,16):
                if _time_left() < 5.0:
                    break
                if not adj[i,j]:
                    adj[i,j] = adj[j,i] = True
                    new_q3 = count_q3(adj)
                    new_edges = adj.sum() // 2
                    if new_q3 == 0:
                        best_edges = new_edges
                        best_adj = adj.copy()
                        improved = True
                    else:
                        adj[i,j] = adj[j,i] = False
    
    # Local search: try swapping edges
    improved = True
    while improved and _time_left() > 5.0:
        improved = False
        for i in range(16):
            for j in range(i+1,16):
                if _time_left() < 5.0:
                    break
                if adj[i,j]:
                    for k in range(16):
                        for l in range(k+1,16):
                            if _time_left() < 5.0:
                                break
                            if not adj[k,l] and (k,l) != (i,j):
                                adj[i,j] = adj[j,i] = False
                                adj[k,l] = adj[l,k] = True
                                new_q3 = count_q3(adj)
                                new_edges = adj.sum() // 2
                                if new_q3 == 0 and new_edges > best_edges:
                                    best_edges = new_edges
                                    best_adj = adj.copy()
                                    improved = True
                                else:
                                    adj[i,j] = adj[j,i] = True
                                    adj[k,l] = adj[l,k] = False
    
    # Return the best graph
    edges = []
    for i in range(16):
        for j in range(i+1,16):
            if best_adj[i,j]:
                edges.append([i,j])
    
    final_q3 = count_q3(best_adj)
    if final_q3 != 0:
        edges = [[a,b] for a in range(3) for b in range(3,16)]
    
    return edges
