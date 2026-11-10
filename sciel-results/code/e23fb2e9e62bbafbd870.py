import numpy as np
import itertools
import time

def kissing_number11() -> np.ndarray:
    d = 11
    # Generate minimal vectors of D11: +/- e_i +/- e_j for i < j
    vectors = []
    for i in range(d):
        for j in range(i + 1, d):
            vectors.append([0] * d)
            vectors[-1][i] = 1
            vectors[-1][j] = 1
            vectors.append([0] * d)
            vectors[-1][i] = 1
            vectors[-1][j] = -1
            vectors.append([0] * d)
            vectors[-1][i] = -1
            vectors[-1][j] = 1
            vectors.append([0] * d)
            vectors[-1][i] = -1
            vectors[-1][j] = -1
            
    V = np.array(vectors, dtype=np.int64)
    n = len(V)
    
    # Compute dot products
    # D[i, j] = V[i] . V[j]
    D = V @ V.T
    
    # Build adjacency matrix for G: edge if D[i, j] <= 0 and i != j
    # We want a clique in this graph.
    # adj[i, j] = 1 if D[i, j] <= 0
    adj = (D <= 0).astype(np.int8)
    np.fill_diagonal(adj, 0)
    
    # Heuristic to find a large clique
    # We will use a greedy approach with multiple restarts and random perturbations
    best_clique = []
    best_size = 0
    
    # Deadline guard
    start_time = time.time()
    deadline = 600 - 10 # 590s safety margin
    
    # Simple greedy: pick node with max degree in remaining subgraph
    # To improve, we can try to pick nodes that are part of dense subgraphs
    
    def greedy_clique(adj, n, seed=0):
        rng = np.random.default_rng(seed)
        remaining = set(range(n))
        clique = []
        
        while remaining:
            # Pick node with max degree in remaining subgraph
            # To add randomness, we can pick from top-k nodes
            degrees = [sum(adj[u, v] for v in remaining if v != u) for u in remaining]
            # Get indices of top degrees
            # For speed, just pick max, or random from top few
            # Let's pick max degree, tie-break by random
            max_deg = max(degrees)
            candidates = [u for u, deg in zip(remaining, degrees) if deg == max_deg]
            if not candidates:
                break
            v = rng.choice(candidates)
            clique.append(v)
            # Update remaining: keep only neighbors of v
            neighbors = [u for u in remaining if adj[v, u] == 1]
            remaining = set(neighbors)
            
        return clique

    # Try multiple seeds
    for seed in range(100):
        if time.time() > deadline:
            break
        clique = greedy_clique(adj, n, seed)
        if len(clique) > best_size:
            best_size = len(clique)
            best_clique = clique
            
    # If best_clique is empty, fallback to 2 points
    if not best_clique:
         best_clique = [0, 1] # e1+e2 and e1-e2? Dot is 0. Valid.
         # Check if 0 and 1 are valid
         # V[0] = e0+e1, V[1] = e0-e1. Dot = 1-1=0. OK.
         
    points = V[best_clique]
    
    # Verification
    # Check distinctness and distance condition
    k = len(points)
    if k < 2:
        return np.array([[1,0,0,0,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
        
    max_norm_sq = 0
    for p in points:
        n_sq = np.dot(p, p)
        if n_sq > max_norm_sq:
            max_norm_sq = n_sq
            
    min_dist_sq = float('inf')
    for i in range(k):
        for j in range(i+1, k):
            diff = points[i] - points[j]
            d_sq = np.dot(diff, diff)
            if d_sq < min_dist_sq:
                min_dist_sq = d_sq
                
    if min_dist_sq <= max_norm_sq:
        # Fallback to safe 2-point solution
        return np.array([[1,0,0,0,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)

    return points
