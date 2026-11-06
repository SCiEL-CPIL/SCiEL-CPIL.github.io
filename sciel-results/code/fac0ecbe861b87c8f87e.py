"""Optimized Neighbor-Joining with vectorized distance computation."""
import numpy as np


def _compute_distance_matrix(sequences, labels):
    """Compute Jukes-Cantor corrected pairwise distance matrix using vectorized NumPy."""
    n = len(labels)
    seq_list = [sequences[label] for label in labels]
    length = len(seq_list[0])
    
    # Encode sequences: A=0, C=1, G=2, T=3
    base_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    # Vectorized encoding using numpy
    seq_matrix = np.array([[base_to_int[ch] for ch in seq] for seq in seq_list], dtype=np.int8)
    
    # Compute pairwise Hamming distances via broadcasting
    # diff[i,j] = number of positions where seq_i != seq_j
    diff_matrix = (seq_matrix[:, None, :] != seq_matrix[None, :, :]).sum(axis=2).astype(np.float64)
    
    # p = proportion of differences
    p = diff_matrix / length
    
    # Jukes-Cantor correction: d = -0.75 * ln(1 - 4p/3)
    # Cap p to avoid log(0) or log(negative)
    np.clip(p, 0, 0.7499, out=p)
    d = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    
    # Set diagonal to 0
    np.fill_diagonal(d, 0.0)
    
    return d


def _neighbor_joining(d):
    """
    Vectorized Neighbor-Joining algorithm on a distance matrix d of shape (n, n).
    Returns a Newick string.
    """
    n = d.shape[0]
    if n == 1:
        return "t0;"
    if n == 2:
        return "(t0,t1);"
    
    D = d.copy()
    tree = [f"t{i}" for i in range(n)]
    active = list(range(n))
    next_node_id = n
    
    while len(active) > 2:
        n_active = len(active)
        
        # Build sub-matrix of active nodes
        D_sub = D[np.ix_(active, active)]
        
        # Compute row sums r[i] = sum_j D_sub[i, j]
        r = D_sub.sum(axis=1)
        
        # Q matrix: Q[i,j] = (n_active - 2) * D[i,j] - r[i] - r[j]
        # Vectorized: Q = (n-2)*D_sub - r[:,None] - r[None,:]
        Q = (n_active - 2) * D_sub - r[:, None] - r[None, :]
        
        # Set diagonal to inf so we don't pick i==j
        np.fill_diagonal(Q, np.inf)
        
        # Find the pair (i, j) that minimizes Q[i,j]
        # Get the minimum value and its position
        min_val = Q.min()
        # Find first occurrence of min_val
        min_pos = np.argmin(Q)
        i_idx = min_pos // n_active
        j_idx = min_pos % n_active
        
        a = active[i_idx]
        b = active[j_idx]
        
        # Compute branch lengths
        delta = (r[i_idx] - r[j_idx]) / (2.0 * (n_active - 2))
        len_a = 0.5 * D[a, b] + 0.5 * delta
        len_b = 0.5 * D[a, b] - 0.5 * delta
        
        # Ensure non-negative
        if len_a < 0:
            len_a = 0.0
        if len_b < 0:
            len_b = 0.0
        
        # Create new node
        new_label = f"({tree[a]}:{len_a:.4f},{tree[b]}:{len_b:.4f})"
        
        # Remove a and b from active, add new node
        new_active = [x for x in active if x not in (a, b)]
        new_node = next_node_id
        next_node_id += 1
        
        # Compute distances from new node to all other active nodes
        # D[new, x] = 0.5 * (D[a, x] + D[b, x] - D[a, b])
        old_n = D.shape[0]
        new_D = np.zeros((old_n + 1, old_n + 1))
        new_D[:old_n, :old_n] = D
        
        # Vectorized update for new distances
        for x in new_active:
            new_D[new_node, x] = 0.5 * (D[a, x] + D[b, x] - D[a, b])
            new_D[x, new_node] = new_D[new_node, x]
        
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
