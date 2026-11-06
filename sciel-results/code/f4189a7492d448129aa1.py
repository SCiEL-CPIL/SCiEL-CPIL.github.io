"""Optimized Neighbor-Joining with vectorized distance computation."""
import numpy as np
import math


def _compute_distance_matrix(sequences, labels):
    """Compute Jukes-Cantor corrected pairwise distance matrix using efficient bit-packing."""
    n = len(labels)
    seq_list = [sequences[label] for label in labels]
    length = len(seq_list[0])
    
    # Encode each base as 2 bits: A=00, C=01, G=10, T=11
    # Pack sequences into integer arrays for fast XOR-based difference counting
    base_to_bits = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    # For each sequence, create an array of 2-bit values
    # We'll use uint8 for simplicity, then compute differences
    
    # Build matrix of shape (n, length) with values 0-3
    seq_matrix = np.zeros((n, length), dtype=np.uint8)
    for i, seq in enumerate(seq_list):
        for j, ch in enumerate(seq):
            seq_matrix[i, j] = base_to_bits[ch]
    
    # Compute pairwise Hamming distances using broadcasting
    # diff_matrix[i, j] = number of positions where seq_matrix[i, k] != seq_matrix[j, k]
    # Using: (a != b).sum() is efficient with NumPy
    diff_matrix = (seq_matrix[:, None, :] != seq_matrix[None, :, :]).sum(axis=2).astype(np.float64)
    
    # Compute p = diff / length
    p = diff_matrix / length
    
    # Jukes-Cantor correction: d = -0.75 * ln(1 - 4p/3)
    # Handle p >= 0.75 (saturation) by capping
    np.clip(p, 0, 0.7499, out=p)
    d = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    
    # Set diagonal to 0
    np.fill_diagonal(d, 0.0)
    
    return d


def _neighbor_joining(d):
    """
    Neighbor-Joining algorithm on a distance matrix d of shape (n, n).
    Returns a Newick string.
    Uses vectorized operations for row sums and Q-matrix computation.
    """
    n = d.shape[0]
    if n == 1:
        return "t0;"
    if n == 2:
        return "(t0,t1);"
    
    # Work with a copy
    D = d.copy()
    
    # Track the current tree structure for each node
    tree = [f"t{i}" for i in range(n)]
    
    # Active nodes
    active = list(range(n))
    new_node_counter = n
    
    while len(active) > 2:
        n_active = len(active)
        
        # Build sub-matrix of active nodes for vectorized computation
        active_arr = np.array(active)
        sub_D = D[np.ix_(active_arr, active_arr)]
        
        # Compute row sums r[i] = sum_j D[i, j] for i in active (excluding diagonal)
        r = sub_D.sum(axis=1)
        
        # Find the pair (a, b) that minimizes:
        # Q(a, b) = (n_active - 2) * D[a, b] - r[a] - r[b]
        # Vectorized: Q[i, j] = (n_active - 2) * sub_D[i, j] - r[i] - r[j]
        Q = (n_active - 2) * sub_D - r[:, None] - r[None, :]
        
        # Set diagonal to inf so we don't pick it
        np.fill_diagonal(Q, float('inf'))
        
        # Find the minimum Q value (upper triangle)
        # Use argmin on the full matrix, but we need to avoid diagonal
        # Flatten and find min, then convert back
        q_flat = Q.flatten()
        min_idx = np.argmin(q_flat)
        best_i = min_idx // n_active
        best_j = min_idx % n_active
        
        # Ensure i < j for consistency
        if best_i > best_j:
            best_i, best_j = best_j, best_i
        
        a = active[best_i]
        b = active[best_j]
        
        # Compute branch lengths
        delta = (r[best_i] - r[best_j]) / (2.0 * (n_active - 2))
        len_a = 0.5 * D[a, b] + 0.5 * delta
        len_b = 0.5 * D[a, b] - 0.5 * delta
        
        # Ensure non-negative
        len_a = max(0.0, len_a)
        len_b = max(0.0, len_b)
        
        # Create new node
        new_label = f"({tree[a]}:{len_a:.4f},{tree[b]}:{len_b:.4f})"
        
        # Create new active list without a and b, add new node
        new_active = [x for x in active if x not in (a, b)]
        new_node = new_node_counter
        new_node_counter += 1
        
        # Compute distances from new node to all other active nodes
        # D[new, x] = 0.5 * (D[a, x] + D[b, x] - D[a, b])
        # Vectorized computation
        other_active_arr = np.array(new_active)
        if len(other_active_arr) > 0:
            new_dists = 0.5 * (D[a, other_active_arr] + D[b, other_active_arr] - D[a, b])
        else:
            new_dists = np.array([])
        
        # Extend D to include new node
        old_n = D.shape[0]
        new_D = np.zeros((old_n + 1, old_n + 1))
        new_D[:old_n, :old_n] = D
        
        if len(other_active_arr) > 0:
            new_D[new_node, other_active_arr] = new_dists
            new_D[other_active_arr, new_node] = new_dists
        
        D = new_D
        tree.append(new_label)
        active = new_active + [new_node]
    
    # Combine the last two nodes
    if len(active) == 2:
        i, j = active
        d_ij = D[i, j]
        half = d_ij / 2.0
        newick = f"({tree[i]}:{half:.4f},{tree[j]}:{half:.4f})"
    else:
        newick = tree[active[0]]
    
    return newick + ";"


def reconstruct(sequences):
    """
    Reconstruct unrooted tree topology from DNA sequences.
    
    Args:
        sequences: dict mapping taxon labels (t0, t1, ...) to DNA strings
        
    Returns:
        Newick string
    """
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] + ";" if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    # Compute distance matrix
    D = _compute_distance_matrix(sequences, labels)
    
    # Run Neighbor-Joining
    newick = _neighbor_joining(D)
    
    return newick
