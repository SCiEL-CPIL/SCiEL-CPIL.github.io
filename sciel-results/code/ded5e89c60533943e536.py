import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # We know n=47 is impossible. Target n=46.
    # Requirement: min_degree >= 46 - 39 = 7.
    
    n = 46
    target_min_deg = 7
    
    best_result = {'n': 39, 'edges': []} # Fallback
    
    # Try to find a valid graph for n=46
    found = False
    
    # Multiple restarts with different random seeds
    for seed in range(500):
        if time.time() > deadline:
            break
            
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize with random graph
        # Expected degree ~ 10 to allow room for deletion
        p = 10.0 / n
        A = (np.random.rand(n, n) < p).astype(int)
        np.fill_diagonal(A, 0)
        # Symmetrize
        A = ((A + A.T) > 0).astype(int)
        
        degrees = A.sum(axis=1)
        
        # Deletion loop
        # We want to eliminate all C4s.
        for _ in range(1000): # Max iterations
            if time.time() > deadline:
                break
            
            # Check for C4s
            common = A @ A
            mask = (common >= 2)
            np.fill_diagonal(mask, False)
            
            if not np.any(mask):
                # No C4s found
                min_deg = np.min(degrees)
                if min_deg >= target_min_deg:
                    # Success
                    edges = []
                    for i in range(n):
                        for j in range(i+1, n):
                            if A[i,j]:
                                edges.append([i, j])
                    best_result = {'n': n, 'edges': edges}
                    found = True
                    break
                else:
                    # Min degree too low, this attempt failed
                    break
            
            # Pick a random pair (i, j) from the mask
            indices = np.argwhere(mask)
            if len(indices) == 0:
                break
            i, j = indices[random.randint(0, len(indices)-1)]
            
            # Find common neighbors
            common_neighbors = np.where((A[i] & A[j]) > 0)[0]
            if len(common_neighbors) < 2:
                continue
            
            # Pick two
            random.shuffle(common_neighbors)
            w = common_neighbors[0]
            x = common_neighbors[1]
            
            # The C4 is i-w-j-x-i
            candidates = [(i, w), (w, j), (j, x), (x, i)]
            
            # Pick the edge to remove
            # Heuristic: remove edge incident to vertices with higher degrees
            best_edge = None
            best_score = -1
            for u, v in candidates:
                score = min(degrees[u], degrees[v])
                if score > best_score:
                    best_score = score
                    best_edge = (u, v)
            
            if best_edge:
                u, v = best_edge
                A[u, v] = 0
                A[v, u] = 0
                degrees[u] -= 1
                degrees[v] -= 1
        
        if found:
            break
    
    # Verification of the best result
    if best_result['n'] > 39:
        n_res = best_result['n']
        edges = best_result['edges']
        adj = [set() for _ in range(n_res)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        min_deg = min(len(adj[i]) for i in range(n_res))
        if min_deg < n_res - 39:
            best_result = {'n': 39, 'edges': []}
            return best_result
            
        # Check C4
        has_c4 = False
        for u in range(n_res):
            for v in range(u+1, n_res):
                if len(adj[u] & adj[v]) >= 2:
                    has_c4 = True
                    break
            if has_c4:
                break
        
        if has_c4:
            best_result = {'n': 39, 'edges': []}
            
    return best_result
