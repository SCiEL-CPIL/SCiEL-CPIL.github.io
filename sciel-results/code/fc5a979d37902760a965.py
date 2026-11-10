# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates.
    Uses a heuristic to find a large independent set of minimal vectors of the D11 lattice (scaled by 2).
    """
    d = 11
    start_time = time.time()
    deadline = 600.0
    
    # Generate minimal vectors of D11 scaled by 2
    # D11 minimal vectors are +/- e_i +/- e_j for i != j
    # Scaled by 2: coordinates are +/- 2
    vectors = []
    for i in range(d):
        for j in range(i + 1, d):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = 2 * s1
                    v[j] = 2 * s2
                    vectors.append(v)
    
    n_vecs = len(vectors)
    
    # Compute dot products
    # Matrix M where M[i,j] = vectors[i] . vectors[j]
    # We only care if M[i,j] > 0
    # Since n_vecs is 220, we can store the conflict graph as a list of sets
    conflicts = [set() for _ in range(n_vecs)]
    
    # Optimization: Compute dot products on the fly or use matrix multiplication
    # V = np.array(vectors)
    # D = V @ V.T
    # But we need to build the graph.
    
    V = np.array(vectors)
    D = V @ V.T
    
    for i in range(n_vecs):
        for j in range(i + 1, n_vecs):
            if D[i, j] > 0:
                conflicts[i].add(j)
                conflicts[j].add(i)
                
    # Heuristic: Minimum Degree Independent Set
    # active_nodes: set of indices not yet removed
    active_nodes = set(range(n_vecs))
    independent_set = []
    
    # To make it faster, we can pre-compute degrees and update
    # But n=220 is small.
    
    while active_nodes:
        # Find node with minimum degree in the induced subgraph on active_nodes
        min_deg = float('inf')
        min_node = -1
        for u in active_nodes:
            deg = len([v for v in conflicts[u] if v in active_nodes])
            if deg < min_deg:
                min_deg = deg
                min_node = u
            if time.time() > deadline - 60:
                break
        
        if min_node == -1:
            break
            
        independent_set.append(min_node)
        
        # Remove min_node and its neighbors from active_nodes
        to_remove = set(conflicts[min_node])
        to_remove.add(min_node)
        active_nodes -= to_remove
        
    # Convert indices to points
    points = np.array([vectors[i] for i in independent_set], dtype=np.int64)
    
    # Validation
    if points.shape[1] != 11:
        return np.array([np.ones(11), -1 * np.ones(11)]).astype(np.int64)
    if len(points) < 2:
        return np.array([np.ones(11), -1 * np.ones(11)]).astype(np.int64)
        
    # Check no zero vector
    if np.any(np.all(points == 0, axis=1)):
        return np.array([np.ones(11), -1 * np.ones(11)]).astype(np.int64)
        
    # Check min dist > max norm
    max_norm = np.max(np.sum(points**2, axis=1))
    min_dist = float('inf')
    k = len(points)
    for i in range(k):
        for j in range(i + 1, k):
            diff = points[i] - points[j]
            dist_sq = np.sum(diff**2)
            if dist_sq < min_dist:
                min_dist = dist_sq
                
    if min_dist <= max_norm:
        # Fallback to seed if heuristic failed constraints (should not happen for independent set of D11 scaled)
        return np.array([np.ones(11), -1 * np.ones(11)]).astype(np.int64)
        
    return points

# EVOLVE-BLOCK-END
