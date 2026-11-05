import sys
import time
import math
import numpy as np

sys.setrecursionlimit(10000)

# Deadline guard: 270s wall-clock limit, use 250s as safety margin
_DEADLINE = time.time() + 250.0


def _time_exceeded():
    return time.time() > _DEADLINE


def _jc_distance_matrix(sequences, labels):
    """Compute pairwise Jukes-Cantor corrected distance matrix using vectorized NumPy."""
    n = len(labels)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    
    seqs = [sequences[label] for label in labels]
    L = len(seqs[0]) if seqs else 0
    
    if L == 0:
        return np.zeros((n, n), dtype=np.float64)
    
    # Convert sequences to integer arrays
    lookup = np.zeros(256, dtype=np.int8)
    lookup[65] = 0  # A
    lookup[67] = 1  # C
    lookup[71] = 2  # G
    lookup[84] = 3  # T
    lookup[97] = 0  # a
    lookup[99] = 1  # c
    lookup[103] = 2  # g
    lookup[116] = 3  # t
    
    seq_arrays = np.zeros((n, L), dtype=np.int8)
    for i, s in enumerate(seqs):
        seq_arrays[i] = lookup[np.frombuffer(s.encode('ascii'), dtype=np.uint8)]
    
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        diffs = np.count_nonzero(seq_arrays[i+1:] != seq_arrays[i], axis=1)
        p = diffs / L
        p_clipped = np.minimum(p, 0.7499)
        d = -0.75 * np.log(1.0 - 4.0 * p_clipped / 3.0)
        mask = p >= 0.7499
        d[mask] = 3.0
        D[i, i+1:] = d
        D[i+1:, i] = d
    return D


def _neighbor_joining(D, labels):
    """
    Perform Neighbor Joining on distance matrix D.
    Returns a nested list representation of the tree.
    """
    n = len(labels)
    if n <= 1:
        return labels[0]
    if n == 2:
        return [labels[0], labels[1]]
    
    dist = D.copy()
    names = list(labels)
    tree_parts = list(names)
    
    num_nodes = n
    while num_nodes > 3:
        if _time_exceeded():
            break
        
        m = num_nodes
        r = np.sum(dist, axis=1)
        criteria = (m - 2) * dist - r[:, np.newaxis] - r[np.newaxis, :]
        np.fill_diagonal(criteria, np.inf)
        iu = np.triu_indices(m, k=1)
        flat_criteria = criteria[iu]
        best_idx = np.argmin(flat_criteria)
        best_i, best_j = iu[0][best_idx], iu[1][best_idx]
        
        i, j = best_i, best_j
        
        new_part = [tree_parts[i], tree_parts[j]]
        
        keep_mask = np.ones(m, dtype=bool)
        keep_mask[i] = False
        keep_mask[j] = False
        new_active = np.where(keep_mask)[0]
        
        new_distances = 0.5 * (dist[i, new_active] + dist[j, new_active] - dist[i, j])
        
        new_dist = np.empty((m - 1, m - 1), dtype=np.float64)
        new_dist[:m-2, :m-2] = dist[np.ix_(new_active, new_active)]
        new_dist[:m-2, m-2] = new_distances
        new_dist[m-2, :m-2] = new_distances
        new_dist[m-2, m-2] = 0.0
        
        tree_parts = [tree_parts[k] for k in new_active] + [new_part]
        dist = new_dist
        num_nodes = m - 1
    
    if num_nodes == 3:
        result = [tree_parts[0], [tree_parts[1], tree_parts[2]]]
    else:
        result = tree_parts[0] if num_nodes == 1 else [tree_parts[0], tree_parts[1]]
        
    return result


def _tree_to_newick(tree):
    """Convert nested list tree to Newick string."""
    if isinstance(tree, str):
        return tree
    else:
        return f"({_tree_to_newick(tree[0])},{_tree_to_newick(tree[1])})"


def reconstruct(sequences):
    """
    Reconstruct the unrooted tree topology from sequences.
    Returns a Newick string over labels t0..t(n-1).
    """
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] + ";"
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"
    
    # Map labels to indices 0..n-1
    label_to_idx = {label: i for i, label in enumerate(labels)}
    
    # Compute distance matrix
    D = _jc_distance_matrix(sequences, labels)
    
    # Run Neighbor Joining
    tree = _neighbor_joining(D, labels)
    
    # Convert tree to Newick
    newick = _tree_to_newick(tree) + ";"
    
    return newick
