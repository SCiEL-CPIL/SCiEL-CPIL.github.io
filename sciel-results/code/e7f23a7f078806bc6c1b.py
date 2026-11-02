"""Rewritten program using a greedy heuristic to find a large kissing configuration in R^11."""
import itertools
import numpy as np

D = 11

# EVOLVE-BLOCK-START
def construct():
    # Generate candidates: Weight 1, 2, 3
    candidates = []
    indices = range(D)
    
    # Weight 1
    for i in indices:
        for s in (1, -1):
            v = [0] * D
            v[i] = s
            candidates.append(v)
            
    # Weight 2
    for i, j in itertools.combinations(indices, 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                candidates.append(v)
                
    # Weight 3
    for i, j, k in itertools.combinations(indices, 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0] * D
                    v[i], v[j], v[k] = si, sj, sk
                    candidates.append(v)
                    
    n = len(candidates)
    norms2 = np.array([sum(x*x for x in v) for v in candidates])
    
    # Build conflict graph adjacency sets
    adj = [set() for _ in range(n)]
    for i in range(n):
        ui = np.array(candidates[i])
        ni = norms2[i]
        for j in range(i + 1, n):
            uj = np.array(candidates[j])
            nj = norms2[j]
            dot = np.dot(ui, uj)
            dist2 = ni + nj - 2 * dot
            max_norm = max(ni, nj)
            if dist2 < max_norm:
                adj[i].add(j)
                adj[j].add(i)
                
    # Greedy MIS: Min-degree heuristic
    remaining = set(range(n))
    mis = []
    
    while remaining:
        min_deg = float('inf')
        best_node = -1
        for u in remaining:
            deg = len(adj[u] & remaining)
            if deg < min_deg:
                min_deg = deg
                best_node = u
                if deg == 0:
                    break
        if best_node == -1:
            break
        mis.append(best_node)
        to_remove = {best_node} | (adj[best_node] & remaining)
        remaining -= to_remove
        
    return np.array([candidates[i] for i in mis], dtype=np.int64)

# EVOLVE-BLOCK-END
