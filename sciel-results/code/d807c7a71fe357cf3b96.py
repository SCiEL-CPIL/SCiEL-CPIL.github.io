"""Optimized Neighbor-Joining with vectorized distance computation."""
import numpy as np
import math


def _compute_distance_matrix(sequences, labels):
    """Compute Jukes-Cantor corrected pairwise distance matrix using NumPy vectorization."""
    n = len(labels)
    seq_list = [sequences[label] for label in labels]
    length = len(seq_list[0])
    
    # Convert to binary matrix: 1 if A, 0 otherwise (we'll do this for each base)
    # Actually, for efficiency, we can compute differences directly
    # Let's encode each sequence as a numpy array of integers
    # A=0, C=1, G=2, T=3
    base_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    # Create matrix of shape (n, length)
    seq_matrix = np.zeros((n, length), dtype=np.int8)
    for i, seq in enumerate(seq_list):
        for j, ch in enumerate(seq):
            seq_matrix[i, j] = base_to_int[ch]
    
    # Compute pairwise differences
    # For each pair (i, j), count positions where seq_matrix[i] != seq_matrix[j]
    # Use broadcasting: diff[i, j, k] = seq_matrix[i, k] != seq_matrix[j, k]
    # This would be (n, n, length) which for n=32, length=2000 is 32*32*2000 = 2M entries, fine
    
    # More efficient: compute sum of differences
    # diff_matrix[i, j] = sum over k of (seq_matrix[i, k] != seq_matrix[j, k])
    
    # Use: (seq_matrix[:, None, :] != seq_matrix[None, :, :]).sum(axis=2)
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
    
    # Work with a copy
    D = d.copy()
    
    # Track the current tree structure for each node
    # tree[i] is the Newick substring for node i
    tree = [f"t{i}" for i in range(n)]
    
    # Active nodes
    active = list(range(n))
    
    # We'll maintain D as a numpy array and update it as we join
    # For efficiency with small n (<=32), we can use a simple approach
    
    current_n = n
    D = D.copy()
    tree = [f"t{i}" for i in range(n)]
    active = list(range(n))
    
    # Map from active index to original index for tracking
    # Actually, let's just track which original taxa each current node represents
    
    while len(active) > 2:
        n_active = len(active)
        
        # Compute row sums r[i] = sum_j D[i, j] for i in active
        # Build a sub-matrix of active nodes
        # For small n, just do it directly
        
        # r[i] for i in active
        r = np.zeros(n_active)
        for i_idx, i in enumerate(active):
            for j_idx, j in enumerate(active):
                if i != j:
                    r[i_idx] += D[i, j]
        
        # Find the pair (a, b) that minimizes:
        # Q(a, b) = (n_active - 2) * D[a, b] - r[a] - r[b]
        
        best_q = float('inf')
        best_i = -1
        best_j = -1
        
        for i_idx in range(n_active):
            for j_idx in range(i_idx + 1, n_active):
                i = active[i_idx]
                j = active[j_idx]
                q = (n_active - 2) * D[i, j] - r[i_idx] - r[j_idx]
                if q < best_q:
                    best_q = q
                    best_i = i_idx
                    best_j = j_idx
        
        a = active[best_i]
        b = active[best_j]
        
        # Compute branch lengths
        # delta = (r[a] - r[b]) / (2 * (n_active - 2))
        # len_a = 0.5 * D[a, b] + 0.5 * delta
        # len_b = 0.5 * D[a, b] - 0.5 * delta
        
        if n_active > 2:
            delta = (r[best_i] - r[best_j]) / (2.0 * (n_active - 2))
        else:
            delta = 0.0
        len_a = 0.5 * D[a, b] + 0.5 * delta
        len_b = 0.5 * D[a, b] - 0.5 * delta
        
        # Ensure non-negative
        if len_a < 0:
            len_a = 0.0
        if len_b < 0:
            len_b = 0.0
        
        # Create new node
        new_label = f"({tree[a]}:{len_a:.4f},{tree[b]}:{len_b:.4f})"
        
        # Find indices of a and b in active
        a_pos = active.index(a)
        b_pos = active.index(b)
        
        # Create new active list without a and b, add new node
        new_active = [x for x in active if x not in (a, b)]
        new_node = max(active) + 1  # Use a new index
        
        # Compute distances from new node to all other active nodes
        # D[new, x] = 0.5 * (D[a, x] + D[b, x] - D[a, b])
        
        # Extend D to include new node
        old_n = D.shape[0]
        new_D = np.zeros((old_n + 1, old_n + 1))
        new_D[:old_n, :old_n] = D
        
        for x in new_active:
            new_D[new_node, x] = 0.5 * (D[a, x] + D[b, x] - D[a, b])
            new_D[x, new_node] = new_D[new_node, x]
        
        D = new_D
        tree.append(new_label)
        active = new_active + [new_node]
    
    # Combine the last two nodes
    if len(active) == 2:
        i, j = active
        # The remaining edge length: we can compute it or just connect them
        # For topology only, we just need to connect them
        d_ij = D[i, j]
        # Split the distance equally or just use it as a single branch
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
