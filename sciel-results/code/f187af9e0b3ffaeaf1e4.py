"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import numpy as np

def construct():
    random.seed(42)
    
    for n in [46, 45, 44, 43, 42, 41, 40, 39]:
        min_deg = n - 39
        
        for attempt in range(3):
            p = min_deg / (n - 1)
            adj = np.zeros((n, n), dtype=np.int8)
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < p:
                        adj[i][j] = 1
                        adj[j][i] = 1
            
            # Ensure min degree
            degs = adj.sum(axis=1)
            for i in range(n):
                while degs[i] < min_deg:
                    candidates = np.where(adj[i] == 0)[0]
                    candidates = candidates[candidates != i]
                    if len(candidates) == 0:
                        break
                    j = random.choice(candidates)
                    adj[i][j] = 1
                    adj[j][i] = 1
                    degs[i] += 1
                    degs[j] += 1
            
            # Eliminate C4s
            for _ in range(100):
                found = False
                for i in range(n):
                    for j in range(i+1, n):
                        common = np.where(adj[i] & adj[j])[0]
                        common = common[(common != i) & (common != j)]
                        if len(common) >= 2:
                            degs = adj.sum(axis=1)
                            removed = False
                            for k in common:
                                if degs[i] > min_deg:
                                    adj[i][k] = 0
                                    adj[k][i] = 0
                                    removed = True
                                    break
                                if degs[j] > min_deg:
                                    adj[j][k] = 0
                                    adj[k][j] = 0
                                    removed = True
                                    break
                            if removed:
                                found = True
                                break
                    if found:
                        break
                if not found:
                    break
            
            degs = adj.sum(axis=1)
            if int(np.min(degs)) >= min_deg:
                valid = True
                for i in range(n):
                    for j in range(i+1, n):
                        common = np.where(adj[i] & adj[j])[0]
                        common = common[(common != i) & (common != j)]
                        if len(common) > 1:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    edges = []
                    for i in range(n):
                        for j in range(i+1, n):
                            if adj[i][j]:
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
    
    return {'n': 39, 'edges': []}
# EVOLVE-BLOCK-END

