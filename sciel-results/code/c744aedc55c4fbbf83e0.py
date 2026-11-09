"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    min_verts = P['minimum_vertices']
    
    # Build adjacency matrix as a 2D numpy array of 0/1
    adj_mat = np.zeros((n, n), dtype=np.uint8)
    for u, v in P['edges']:
        adj_mat[u, v] = 1
        adj_mat[v, u] = 1
    
    # Precompute all subsets as masks with size >= min_verts
    # We'll process them in batches using vectorized operations
    
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_score = -1
    best_mask = 0
    
    # Generate all valid masks (size >= min_verts)
    all_masks = []
    for k in range(min_verts, n + 1):
        for combo in combinations(range(n), k):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            all_masks.append(mask)
    
    # Process in batches to use vectorized numpy operations
    batch_size = 10000
    
    for batch_start in range(0, len(all_masks), batch_size):
        if time.time() > deadline:
            break
        
        batch = all_masks[batch_start:batch_start + batch_size]
        if not batch:
            break
        
        # Convert batch of masks to numpy array of shape (batch_len, n)
        # Each row is the indicator vector of the subset
        masks_arr = np.array(batch, dtype=np.int32)
        
        # For each mask, compute the degree of each vertex in the induced subgraph
        # degree[v] for vertex v in subset S = sum over u in S of adj_mat[v, u]
        # We need: for each mask m, for each vertex v in m, deg = popcount(adj_mask[v] & m)
        
        # Vectorized approach:
        # Create a 3D array: for each mask, for each vertex, is it in the mask?
        # mask_bits[i, v] = 1 if vertex v is in mask i
        mask_bits = np.zeros((len(batch), n), dtype=np.uint8)
        for i in range(len(batch)):
            m = batch[i]
            for v in range(n):
                if m & (1 << v):
                    mask_bits[i, v] = 1
        
        # degrees[i, v] = sum over u of adj_mat[v, u] * mask_bits[i, u]
        # = adj_mat @ mask_bits[i] for each i
        # This is a matrix multiplication: degrees = mask_bits @ adj_mat.T
        # But we only care about vertices v that are IN the mask
        degrees = mask_bits.astype(np.int32) @ adj_mat.T
        
        # For each subset i, distinct degrees among vertices in the subset
        # We need: for each row i, look at degrees[i, v] for v where mask_bits[i, v] == 1
        # and count distinct values
        
        scores = np.zeros(len(batch), dtype=np.int32)
        for i in range(len(batch)):
            if time.time() > deadline:
                break
            # Get degrees of vertices in this subset
            subset_degrees = degrees[i][mask_bits[i] == 1]
            score = len(set(subset_degrees.tolist()))
            scores[i] = score
        
        # Find the best in this batch
        max_idx = np.argmax(scores)
        if scores[max_idx] > best_score:
            best_score = int(scores[max_idx])
            best_mask = batch[max_idx]
            
            if best_score == n:
                break
    
    # Convert best_mask to a list of vertex indices
    vertices = []
    m = best_mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        vertices.append(int(v))
        m ^= lsb
        
    if not vertices:
        return list(range(min_verts))
        
    return vertices
# EVOLVE-BLOCK-END
