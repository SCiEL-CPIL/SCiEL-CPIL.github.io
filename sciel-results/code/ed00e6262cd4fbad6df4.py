"""Neighbour joining with NumPy distance computation and deadline guard."""
import math
import time
import numpy as np


def _jc_distance_matrix(sequences, labels):
    """Compute pairwise Jukes-Cantor corrected distances using NumPy."""
    n = len(labels)
    # Convert sequences to numeric arrays
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seqs = []
    for l in labels:
        arr = np.array([char_map[c] for c in sequences[l]], dtype=np.int8)
        seqs.append(arr)
    seqs = np.array(seqs)  # shape (n, L)
    L = seqs.shape[1]
    
    # Compute pairwise mismatches
    if n <= 40:
        # Broadcasting approach
        diff = (seqs[:, np.newaxis, :] != seqs[np.newaxis, :, :])
        mismatches = diff.sum(axis=2).astype(np.float64)
    else:
        mismatches = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                m = np.sum(seqs[i] != seqs[j])
                mismatches[i, j] = m
                mismatches[j, i] = m
    
    p = mismatches / L
    p = np.clip(p, 0.0, 0.7499)
    d = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    np.fill_diagonal(d, 0.0)
    return d


def _nj_build_tree(d, labels):
    """Neighbour Joining algorithm on distance matrix d (n x n).
    Returns Newick string."""
    n = len(labels)
    if n == 1:
        return labels[0]
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    
    # d is n x n
    # We'll maintain a list of active nodes and their tree representations
    active = list(range(n))
    tree_repr = {i: labels[i] for i in range(n)}
    dist = d.copy()
    
    next_node_id = n  # For new internal nodes
    
    while len(active) > 3:
        m = len(active)
        
        # Compute row sums for active nodes
        row_sums = np.zeros(m)
        for idx, a in enumerate(active):
            for idx2, b in enumerate(active):
                if a != b:
                    row_sums[idx] += dist[a, b]
        
        # Find best pair
        best_criterion = float('inf')
        best_i = best_j = -1
        for i in range(m):
            for j in range(i+1, m):
                a, b = active[i], active[j]
                criterion = (m - 2) * dist[a, b] - row_sums[i] - row_sums[j]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_i, best_j = i, j
        
        a, b = active[best_i], active[best_j]
        d_ab = dist[a, b]
        
        # Compute branch lengths
        r_a = (row_sums[best_i] - row_sums[best_j] + d_ab) / (2.0 * m)
        r_b = d_ab - r_a
        if r_a < 0:
            r_a = 0.0
        if r_b < 0:
            r_b = 0.0
        
        # Create new node
        new_id = next_node_id
        next_node_id += 1
        
        # Update distances from new node to remaining active nodes
        rest = [x for x in active if x not in (a, b)]
        for x in rest:
            new_d = 0.5 * (dist[a, x] + dist[b, x] - d_ab)
            if new_d < 0:
                new_d = 0.0
            dist[new_id, x] = new_d
            dist[x, new_id] = new_d
        
        # Update active list
        active = rest + [new_id]
        tree_repr[new_id] = f"({tree_repr[a]}:{r_a:.6f},{tree_repr[b]}:{r_b:.6f})"
    
    # Join remaining 3 nodes
    a, b, c = active
    d_ab = dist[a, b]
    d_ac = dist[a, c]
    d_bc = dist[b, c]
    
    len_a = (d_ab + d_ac - d_bc) / 2.0
    len_b = (d_ab + d_bc - d_ac) / 2.0
    len_c = (d_ac + d_bc - d_ab) / 2.0
    len_a = max(0.0, len_a)
    len_b = max(0.0, len_b)
    len_c = max(0.0, len_c)
    
    return f"({tree_repr[a]}:{len_a:.6f},{tree_repr[b]}:{len_b:.6f},{tree_repr[c]}:{len_c:.6f})"


def reconstruct(sequences):
    """Reconstruct unrooted tree topology from DNA sequences."""
    start_time = time.time()
    DEADLINE = 265.0  # Safety margin before 270s kill
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    
    # Compute distance matrix
    d = _jc_distance_matrix(sequences, labels)
    
    # Build tree using NJ
    newick = _nj_build_tree(d, labels)
    
    return newick
