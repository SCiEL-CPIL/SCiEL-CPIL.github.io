"""Optimized Neighbor-Joining with vectorized distance computation."""
import numpy as np
import math


def _compute_distance_matrix(sequences, labels):
    """Compute Jukes-Cantor corrected pairwise distance matrix using NumPy vectorization."""
    n = len(labels)
    seq_list = [sequences[label] for label in labels]
    length = len(seq_list[0])
    
    if length == 0:
        return np.zeros((n, n))

    # Efficient encoding of sequences
    # Map ASCII codes of A, C, G, T to 0, 1, 2, 3
    mapping = np.zeros(256, dtype=np.int8)
    mapping[ord('A')] = 0
    mapping[ord('C')] = 1
    mapping[ord('G')] = 2
    mapping[ord('T')] = 3
    
    # Convert list of strings to a 2D numpy array of bytes
    seq_matrix = np.empty((n, length), dtype=np.int8)
    for i, seq in enumerate(seq_list):
        seq_bytes = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)
        seq_matrix[i] = mapping[seq_bytes]
        
    # Compute pairwise differences
    # diff_matrix[i, j] = sum over k of (seq_matrix[i, k] != seq_matrix[j, k])
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
    """
    n = d.shape[0]
    if n == 1:
        return "t0;"
    if n == 2:
        return "(t0,t1);"
    
    D = d.copy()
    tree_labels = [f"t{i}" for i in range(n)]
    active = list(range(n))
    
    while len(active) > 2:
        m = len(active)
        D_sub = D[np.ix_(active, active)]
        r = D_sub.sum(axis=1)
        
        Q = (m - 2) * D_sub - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        
        min_idx_flat = np.argmin(Q)
        i_idx, j_idx = np.unravel_index(min_idx_flat, Q.shape)
        
        if i_idx > j_idx:
            i_idx, j_idx = j_idx, i_idx
            
        a = active[i_idx]
        b = active[j_idx]
        
        delta = (r[i_idx] - r[j_idx]) / (2.0 * (m - 2))
        len_a = 0.5 * D[a, b] + 0.5 * delta
        len_b = 0.5 * D[a, b] - 0.5 * delta
        
        if len_a < 0:
            len_a = 0.0
        if len_b < 0:
            len_b = 0.0
            
        new_label = f"({tree_labels[a]}:{len_a:.4f},{tree_labels[b]}:{len_b:.4f})"
        
        new_node_idx = D.shape[0]
        
        new_D = np.zeros((D.shape[0] + 1, D.shape[0] + 1))
        new_D[:D.shape[0], :D.shape[0]] = D
        
        for x in range(D.shape[0]):
            if x == a or x == b:
                continue
            dist = 0.5 * (D[a, x] + D[b, x] - D[a, b])
            if dist < 0:
                dist = 0.0
            new_D[new_node_idx, x] = dist
            new_D[x, new_node_idx] = dist
            
        D = new_D
        tree_labels.append(new_label)
        
        new_active = [x for x in active if x not in (a, b)]
        new_active.append(new_node_idx)
        active = new_active
        
    if len(active) == 2:
        i, j = active
        d_ij = D[i, j]
        half = d_ij / 2.0
        if half < 0:
            half = 0.0
        newick = f"({tree_labels[i]}:{half:.4f},{tree_labels[j]}:{half:.4f})"
    else:
        newick = tree_labels[active[0]]
        
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
    
    D = _compute_distance_matrix(sequences, labels)
    newick = _neighbor_joining(D)
    
    return newick
