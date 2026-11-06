"""Optimized Neighbor-Joining with vectorized distance computation."""
import numpy as np
import math


def _compute_distance_matrix(sequences, labels):
    """Compute Jukes-Cantor corrected pairwise distance matrix using NumPy vectorization."""
    n = len(labels)
    seq_list = [sequences[label] for label in labels]
    length = len(seq_list[0])
    
    # Encode sequences: A=0, C=1, G=2, T=3
    # Use lookup table for fast conversion
    lookup = np.array([0, 1, 2, 3], dtype=np.int8)
    # Map ASCII: A=65, C=67, G=71, T=84
    ascii_map = np.zeros(128, dtype=np.int8)
    ascii_map[65] = 0  # A
    ascii_map[67] = 1  # C
    ascii_map[71] = 2  # G
    ascii_map[84] = 3  # T
    
    seq_matrix = np.zeros((n, length), dtype=np.int8)
    for i, seq in enumerate(seq_list):
        # Convert string to int array using lookup
        arr = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)
        seq_matrix[i] = ascii_map[arr]
    
    # Compute pairwise Hamming distances using vectorized operations
    # diff[i,j] = sum_k (seq_matrix[i,k] != seq_matrix[j,k])
    # For n<=32, length<=2000: n*n*length = 32*32*2000 = 2M, fine
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
    Uses vectorized operations for efficiency.
    """
    n = d.shape[0]
    if n == 1:
        return "t0;"
    if n == 2:
        return "(t0,t1);"
    
    # Work with a copy
    D = d.copy()
    tree = [f"t{i}" for i in range(n)]
    active = list(range(n))
    
    # Track which nodes have been removed
    removed = [False] * n
    node_count = n
    
    while len(active) > 2:
        n_active = len(active)
        
        # Get the active sub-matrix
        D_active = D[np.ix_(active, active)]
        
        # Compute row sums r[i] = sum_j D[i, j] for i in active
        # Exclude diagonal (set to 0 first)
        D_active_copy = D_active.copy()
        np.fill_diagonal(D_active_copy, 0.0)
        r = D_active_copy.sum(axis=1)
        
        # Find the pair (a, b) that minimizes Q(a, b) = (n_active - 2) * D[a, b] - r[a] - r[b]
        # Compute Q matrix
        Q = (n_active - 2) * D_active - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)  # Don't pick same node
        
        # Find minimum Q value (upper triangle to avoid double counting)
        i_idx, j_idx = np.unravel_index(np.argmin(Q), Q.shape)
        
        # Ensure i_idx < j_idx
        if i_idx > j_idx:
            i_idx, j_idx = j_idx, i_idx
        
        a = active[i_idx]
        b = active[j_idx]
        
        # Compute branch lengths
        delta = (r[i_idx] - r[j_idx]) / (2.0 * (n_active - 2))
        len_a = 0.5 * D[a, b] + 0.5 * delta
        len_b = 0.5 * D[a, b] - 0.5 * delta
        
        # Ensure non-negative
        len_a = max(0.0, len_a)
        len_b = max(0.0, len_b)
        
        # Create new node label
        new_label = f"({tree[a]}:{len_a:.4f},{tree[b]}:{len_b:.4f})"
        
        # Find positions of a and b in active
        a_pos = active.index(a)
        b_pos = active.index(b)
        
        # Create new active list without a and b
        new_active = [x for x in active if x not in (a, b)]
        
        # New node index
        new_node = node_count
        node_count += 1
        
        # Extend D to include new node
        old_n = D.shape[0]
        new_D = np.zeros((old_n + 1, old_n + 1))
        new_D[:old_n, :old_n] = D
        
        # Compute distances from new node to all other active nodes
        # D[new, x] = 0.5 * (D[a, x] + D[b, x] - D[a, b])
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
        half = max(0.0, half)
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
