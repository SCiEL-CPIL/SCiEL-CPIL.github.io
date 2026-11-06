"""Phylogenetic topology recovery via binary-matrix NJ with optimized O(n^3) loop."""
import numpy as np


def _encode_alignment(sequences, labels):
    """Encode DNA sequences into a binary (one-hot) numpy array of shape (n, 4*L)."""
    n = len(labels)
    L = len(sequences[labels[0]])
    # One-hot: for each position, 4 bits indicating which base
    # Total width = 4 * L
    mat = np.zeros((n, 4 * L), dtype=np.uint8)
    base_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    for i, label in enumerate(labels):
        seq = sequences[label]
        row = mat[i]
        for j, ch in enumerate(seq):
            row[j * 4 + base_idx[ch]] = 1
    return mat


def _compute_distance_matrix(sequences, labels):
    """
    Compute Jukes-Cantor corrected pairwise distances using binary matrix dot products.
    For one-hot encoded sequences, Hamming distance between i and j =
      L - (row_i . row_j)
    """
    n = len(labels)
    L = len(sequences[labels[0]])
    
    # Encode
    mat = _encode_alignment(sequences, labels)  # (n, 4L), uint8
    
    # Compute similarity matrix: S[i,j] = mat[i] . mat[j] = number of matching positions
    mat_f = mat.astype(np.float32)
    S = mat_f @ mat_f.T  # (n, n)
    
    # Hamming distances
    hamming = L - S  # (n, n)
    
    # Jukes-Cantor correction
    p = hamming / L
    np.clip(p, 0.0, 0.7499, out=p)
    D = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    np.fill_diagonal(D, 0.0)
    
    return D.astype(np.float64)


def _neighbor_joining(D):
    """
    Optimized Neighbor-Joining on distance matrix D (n x n).
    Returns Newick string (unrooted).
    Uses O(n^3) with NumPy vectorization for Q-matrix computation.
    """
    n = D.shape[0]
    if n == 1:
        return "t0;"
    if n == 2:
        return "(t0,t1);"
    
    dist = D.copy()  # (n, n)
    node_labels = [f"t{i}" for i in range(n)]
    active = list(range(n))
    
    next_node_id = n  # for new internal nodes
    
    while len(active) > 2:
        m = len(active)
        
        # Get submatrix of active nodes
        idx = np.array(active)
        sub = dist[np.ix_(idx, idx)]  # (m, m)
        
        # Row sums
        r = sub.sum(axis=1)  # (m,)
        
        # Q matrix: Q[i,j] = (m-2)*sub[i,j] - r[i] - r[j]
        Q = (m - 2) * sub - r[:, None] - r[None, :]
        
        # Find min Q off-diagonal
        np.fill_diagonal(Q, np.inf)
        
        flat_idx = np.argmin(Q)
        i_idx = flat_idx // m
        j_idx = flat_idx % m
        
        a = active[i_idx]
        b = active[j_idx]
        
        # Branch lengths
        if m > 2:
            delta = (r[i_idx] - r[j_idx]) / (2.0 * (m - 2))
        else:
            delta = 0.0
        len_a = 0.5 * sub[i_idx, j_idx] + 0.5 * delta
        len_b = 0.5 * sub[i_idx, j_idx] - 0.5 * delta
        len_a = max(0.0, len_a)
        len_b = max(0.0, len_b)
        
        # New label
        new_label = f"({node_labels[a]}:{len_a:.4f},{node_labels[b]}:{len_b:.4f})"
        
        # Create new node
        new_id = next_node_id
        next_node_id += 1
        
        # Extend dist matrix
        old_size = dist.shape[0]
        new_dist = np.zeros((old_size + 1, old_size + 1), dtype=np.float64)
        new_dist[:old_size, :old_size] = dist
        
        # Compute distances from new node to all other active nodes
        for k in range(m):
            if k == i_idx or k == j_idx:
                continue
            x = active[k]
            new_dist[new_id, x] = 0.5 * (dist[a, x] + dist[b, x] - dist[a, b])
            new_dist[x, new_id] = new_dist[new_id, x]
        
        dist = new_dist
        node_labels.append(new_label)
        
        # Update active list
        active.remove(a)
        active.remove(b)
        active.append(new_id)
    
    # Combine last two
    if len(active) == 2:
        i, j = active
        d_ij = dist[i, j]
        half = d_ij / 2.0
        newick = f"({node_labels[i]}:{half:.4f},{node_labels[j]}:{half:.4f})"
    else:
        newick = node_labels[active[0]]
    
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
