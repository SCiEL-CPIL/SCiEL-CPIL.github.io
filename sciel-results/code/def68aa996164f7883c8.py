import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    best_result = {'n': 39, 'edges': []}
    best_n = 39
    
    def is_c4_free(adj, n):
        # Check if graph is C4-free
        # C4 exists if any pair has > 1 common neighbor
        # common = adj @ adj
        # We need off-diagonal entries < 2
        common = adj @ adj
        # Zero out diagonal
        np.fill_diagonal(common, 0)
        return np.all(common < 2)

    def count_c4s(adj, n):
        common = adj @ adj
        np.fill_diagonal(common, 0)
        return np.sum(common >= 2)

    def remove_edge_to_break_c4(adj, n, u, v, w):
        # u-v-w is part of a C4? No, u and v have common neighbor w.
        # We need to remove an edge incident to u or v or w?
        # Removing (u, w) or (v, w) or (u, v) (if edge exists) or (v, u)
        # If we remove (u, w), we break the C4 u-w-v-x-u?
        # Actually, if u and v have common neighbor w, and we remove (u, w),
        # then u and v no longer have w as common neighbor.
        # But they might have other common neighbors.
        # We should remove an edge that is in the most C4s.
        # For simplicity, remove (u, w) or (v, w) randomly.
        if adj[u][w]:
            adj[u][w] = 0
            adj[w][u] = 0
        elif adj[v][w]:
            adj[v][w] = 0
            adj[w][v] = 0

    def try_construct_fast(n, seed):
        random.seed(seed)
        target_deg = n - 39
        
        # Try several random densities
        for p in [0.3, 0.25, 0.2, 0.15]:
            adj = np.zeros((n, n), dtype=np.int8)
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < p:
                        adj[i][j] = 1
                        adj[j][i] = 1
            
            # Check initial min degree
            if np.min(adj.sum(axis=1)) < target_deg:
                continue
            
            # Repair loop
            for step in range(2000):
                if time.time() > deadline:
                    return None
                
                # Find C4s
                common = adj @ adj
                np.fill_diagonal(common, 0)
                
                if np.all(common < 2):
                    if np.min(adj.sum(axis=1)) >= target_deg:
                        edges = []
                        for i in range(n):
                            for j in range(i+1, n):
                                if adj[i][j]:
                                    edges.append([i, j])
                        return {'n': n, 'edges': edges}
                    else:
                        break # Min degree dropped too low
                
                # Find a pair with >= 2 common neighbors
                indices = np.triu_indices(n, k=1)
                vals = common[indices]
                viol_mask = vals >= 2
                if not np.any(viol_mask):
                    break
                
                viol_positions = np.where(viol_mask)[0]
                pos = random.choice(viol_positions)
                u, v = indices[0][pos], indices[1][pos]
                
                # Find common neighbors
                commons = np.where((adj[u] & adj[v]) == 1)[0]
                if len(commons) == 0:
                    continue
                
                w = random.choice(commons.tolist())
                
                # Remove edge (u, w) or (v, w)
                # Prefer removing from higher degree vertex?
                deg_u = adj[u].sum()
                deg_v = adj[v].sum()
                
                if deg_u > deg_v:
                    x = u
                elif deg_v > deg_u:
                    x = v
                else:
                    x = u if random.random() < 0.5 else v
                
                adj[x][w] = 0
                adj[w][x] = 0
                
                if np.min(adj.sum(axis=1)) < target_deg:
                    break
        
        return None

    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        found = False
        # Use more seeds for higher n
        num_seeds = 500 if n == 47 else 1000
        
        for seed in range(num_seeds):
            if time.time() > deadline:
                break
            res = try_construct_fast(n, seed)
            if res is not None:
                if n > best_n:
                    best_n = n
                    best_result = res
                found = True
                if n == 47:
                    return best_result
                break
        
        if found:
            continue
        else:
            # If not found, continue to lower n
            pass
    
    return best_result
