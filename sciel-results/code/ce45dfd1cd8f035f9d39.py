"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
'''Optimized Neighbour Joining with NumPy distance calculation and time guard.'''
import numpy as np
import math
import time


def _jc_distance_matrix(seqs):
    """Compute pairwise Jukes-Cantor distances using NumPy."""
    # seqs is a list of strings
    n = len(seqs)
    L = len(seqs[0])
    if n == 0 or L == 0:
        return np.zeros((n, n))
        
    arr = np.array([list(s) for s in seqs])
    # Compare all pairs
    # Shape: (n, n, L)
    # This might be memory intensive for very large L, but L <= 2000, n <= 32.
    # 32*32*2000 = 2M elements. ~2MB. Safe.
    diff = (arr[:, None, :] != arr[None, :, :])
    hamming = diff.sum(axis=2)
    p = hamming / L
    # Clip to avoid log(0) or negative values inside log
    p = np.clip(p, 0.0, 0.7499)
    dist = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    return dist


def reconstruct(sequences):
    start_time = time.time()
    # Budget is 180s for 20 instances. ~9s per instance.
    # We set a conservative limit of 8s per instance.
    deadline = start_time + 8.0
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n == 1:
        return f"({labels[0]});"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
        
    # Get sequences in label order
    seqs = [sequences[l] for l in labels]
    
    # Compute distance matrix
    dist_matrix = _jc_distance_matrix(seqs)
    
    # Initialize distance dictionary for dynamic nodes
    # dist[(i, j)]
    dist = {}
    for i in range(n):
        for j in range(i + 1, n):
            d = float(dist_matrix[i, j])
            dist[(i, j)] = d
            dist[(j, i)] = d
        dist[(i, i)] = 0.0
            
    active = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    next_id = n
    
    while len(active) > 3:
        # Check time
        if time.time() > deadline:
            # If we run out of time, return a star tree of the remaining active nodes
            # combined with the already built subtrees?
            # This is a valid fallback.
            # The active nodes are either original leaves or internal nodes with subtrees.
            # Connecting them in a star is a valid unrooted tree.
            # It might not be the best topology, but it's better than crashing.
            a, b, c = active[:3]
            # If there are more than 3, we just star them all?
            # Newick allows n-ary nodes.
            # (a, b, c, d, ...) is valid.
            return "(" + ",".join(trees[x] for x in active) + ");"
            
        m = len(active)
        
        # Calculate row sums r_i for active nodes
        r = {i: 0.0 for i in active}
        for i in active:
            for j in active:
                if i != j:
                    r[i] += dist[(i, j)]
                    
        # Find pair (i, j) minimizing Q(i, j)
        best_q = float('inf')
        best_i = -1
        best_j = -1
        
        for idx_i, i in enumerate(active):
            for idx_j, j in enumerate(active):
                if idx_j <= idx_i:
                    continue
                q = (m - 2) * dist[(i, j)] - r[i] - r[j]
                if q < best_q:
                    best_q = q
                    best_i = i
                    best_j = j
                    
        i = best_i
        j = best_j
        new_node = next_id
        next_id += 1
        
        trees[new_node] = f"({trees[i]},{trees[j]})"
        
        # Update distances for new node
        d_ij = dist[(i, j)]
        for k in active:
            if k == i or k == j:
                continue
            d_new_k = 0.5 * (dist[(i, k)] + dist[(j, k)] - d_ij)
            dist[(new_node, k)] = d_new_k
            dist[(k, new_node)] = d_new_k
            
        # Update active list
        active.remove(i)
        active.remove(j)
        active.append(new_node)
        
    # Final 3 nodes
    a, b, c = active
    return f"({trees[a]},{trees[b]},{trees[c]});"
# EVOLVE-BLOCK-END
