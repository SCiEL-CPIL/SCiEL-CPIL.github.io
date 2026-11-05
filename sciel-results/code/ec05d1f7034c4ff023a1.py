import numpy as np
import math
import time

def _jc_distance_matrix(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    seqs = np.array([list(sequences[label]) for label in labels])
    # Convert to integers for comparison
    # A=0, C=1, G=2, T=3
    # Or just compare chars
    
    # Efficient way:
    # Create a matrix of 0/1 for mismatches
    # seqs shape: (n, L)
    # We can broadcast: seqs[:, None, :] vs seqs[None, :, :] -> (n, n, L)
    # This might use too much memory for n=32, L=2000. 32*32*2000 = 2M bools = 2MB. Fine.
    
    mismatches = (seqs[:, None, :] != seqs[None, :, :]).sum(axis=2)
    p = mismatches / seqs.shape[1]
    
    # Avoid log(0)
    # p >= 0.75 -> inf or large number
    # In JC, p=0.75 is the max expected for infinite divergence.
    # If p >= 0.75, set to a large constant, e.g., 10.0
    # Or use np.where
    
    d = np.zeros((n, n))
    mask = p < 0.75
    d[mask] = -0.75 * np.log(1.0 - 4.0 * p[mask] / 3.0)
    d[~mask] = 10.0 # Large penalty
    
    # Symmetrize (should be symmetric already)
    # Set diagonal to 0
    np.fill_diagonal(d, 0.0)
    
    return d, labels

def _neighbor_joining(dist, labels):
    n = len(labels)
    if n == 0:
        return ""
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    active = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    
    dist_dict = {}
    for i in range(n):
        for j in range(i+1, n):
            dist_dict[(i, j)] = dist[i, j]
            dist_dict[(j, i)] = dist[i, j]
    
    next_id = n
    
    while len(active) > 3:
        m = len(active)
        
        # Build m x m distance matrix for active nodes
        d_active = np.zeros((m, m))
        for i_idx, ai in enumerate(active):
            for j_idx, aj in enumerate(active):
                if i_idx != j_idx:
                    d_active[i_idx, j_idx] = dist_dict[(ai, aj)]
        
        # Compute r vector: sum of distances from each node
        r = d_active.sum(axis=1)
        
        # Compute Q matrix: Q[i,j] = (m-2)*d[i,j] - r[i] - r[j]
        Q = (m - 2) * d_active - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        
        # Find minimum Q
        min_flat = np.argmin(Q)
        i_idx, j_idx = np.unravel_index(min_flat, Q.shape)
        if i_idx > j_idx:
            i_idx, j_idx = j_idx, i_idx
        
        best_i = active[i_idx]
        best_j = active[j_idx]
        
        # Create new node
        new_id = next_id
        next_id += 1
        
        tree_i = trees[best_i]
        tree_j = trees[best_j]
        new_tree = f"({tree_i},{tree_j})"
        
        # Compute new distances to other active nodes
        for x in active:
            if x not in (best_i, best_j):
                d_new_x = 0.5 * (dist_dict[(best_i, x)] + dist_dict[(best_j, x)] - dist_dict[(best_i, best_j)])
                dist_dict[(new_id, x)] = d_new_x
                dist_dict[(x, new_id)] = d_new_x
        
        # Remove best_i, best_j from active
        active = [x for x in active if x not in (best_i, best_j)]
        active.append(new_id)
        
        trees[new_id] = new_tree
    
    # Final 3 nodes
    if len(active) == 3:
        a, b, c = active
        return f"({trees[a]},({trees[b]},{trees[c]}));"
    else:
        return "(" + ",".join(trees[x] for x in active) + ");"

def reconstruct(sequences):
    start_time = time.time()
    deadline = 260.0  # Safety margin before 270s kill
    
    d, labels = _jc_distance_matrix(sequences)
    
    # Check time
    if time.time() - start_time > deadline - 5:
        n = len(labels)
        if n <= 2:
            return ",".join(labels) + ";"
        return "(" + ",".join(labels) + ");"
    
    # NJ with vectorized Q-matrix
    tree_newick = _neighbor_joining(d, labels)
    
    return tree_newick
