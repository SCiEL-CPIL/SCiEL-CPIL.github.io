import time
import math
import numpy as np

def _jc_distance_matrix(sequences, labels):
    """Compute the Jukes-Cantor corrected distance matrix using NumPy."""
    n = len(labels)
    # Convert sequences to a numpy array of characters
    # seq_array[i] corresponds to labels[i]
    seq_array = np.array([list(sequences[label]) for label in labels], dtype='S1')
    
    L = seq_array.shape[1]
    if L == 0:
        return np.zeros((n, n))
        
    # Calculate pairwise differences
    # seq_i: (n, 1, L)
    # seq_j: (1, n, L)
    seq_i = seq_array[:, np.newaxis, :]
    seq_j = seq_array[np.newaxis, :, :]
    
    # diff[i, j, k] is True if seq[i][k] != seq[j][k]
    diff = (seq_i != seq_j)
    
    # Count differences for each pair (i, j)
    counts = diff.sum(axis=2)
    
    # Proportion of differences
    p = counts / L
    
    # Jukes-Cantor correction
    # d = -3/4 * ln(1 - 4/3 * p)
    # Handle p >= 0.75 (saturation)
    d = np.full((n, n), 3.0)
    mask = p < 0.75
    
    # Calculate log term only for valid entries
    val = 1.0 - (4.0 / 3.0) * p
    # Avoid log(0) or log(negative)
    # For p < 0.75, 1 - 4/3*p > 0
    # We compute log for all, then mask
    # np.log(0) is -inf, so we handle it carefully
    with np.errstate(divide='ignore', invalid='ignore'):
        log_val = np.log(val)
        
    d[mask] = -0.75 * log_val[mask]
    
    # Diagonal is 0
    np.fill_diagonal(d, 0.0)
    
    return d

def _neighbor_joining(d):
    """
    Perform Neighbor Joining on distance matrix d.
    Returns a Newick string.
    """
    n = len(d)
    labels = [f"t{i}" for i in range(n)]
    
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]});"

    # Master distance matrix, sized for all potential nodes (original + internal)
    max_nodes = 2 * n
    D = np.zeros((max_nodes, max_nodes))
    D[:n, :n] = d
    
    active_nodes = list(range(n))
    tree_parts = {i: labels[i] for i in range(n)}
    next_idx = n
    
    while len(active_nodes) > 3:
        k = len(active_nodes)
        
        # Extract submatrix for active nodes
        idx = np.array(active_nodes)
        sub_d = D[np.ix_(idx, idx)]
        
        # Calculate row sums
        r = sub_d.sum(axis=1)
        
        # Calculate Q matrix
        # Q[i, j] = (k-2)*d[i, j] - r[i] - r[j]
        Q = (k - 2) * sub_d - r[:, np.newaxis] - r[np.newaxis, :]
        
        # Ignore diagonal
        np.fill_diagonal(Q, np.inf)
        
        # Find the pair with minimum Q
        min_idx_flat = np.argmin(Q)
        i_idx, j_idx = np.unravel_index(min_idx_flat, Q.shape)
        
        u = active_nodes[i_idx]
        v = active_nodes[j_idx]
        
        # Create new internal node w
        w = next_idx
        next_idx += 1
        
        # Update distances from w to all other active nodes
        # d[w, x] = 0.5 * (d[u, x] + d[v, x] - d[u, v])
        for x in active_nodes:
            if x == u or x == v:
                continue
            D[w, x] = 0.5 * (D[u, x] + D[v, x] - D[u, v])
            D[x, w] = D[w, x]
        D[w, w] = 0.0
        
        # Update active nodes
        # Remove u and v, add w
        active_nodes = [x for x in active_nodes if x not in (u, v)]
        active_nodes.append(w)
        
        # Update tree part for w
        tree_parts[w] = f"({tree_parts[u]},{tree_parts[v]})"
        
    # Final step: connect the remaining 3 nodes
    final_nodes = active_nodes
    if len(final_nodes) == 3:
        newick = f"({tree_parts[final_nodes[0]]},{tree_parts[final_nodes[1]]},{tree_parts[final_nodes[2]]});"
    else:
        # Fallback
        parts = [tree_parts[node] for node in final_nodes]
        newick = "(" + ",".join(parts) + ");"
        
    return newick

def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 270.0
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] if n == 1 else ""
    
    # Compute distance matrix
    d = _jc_distance_matrix(sequences, labels)
    
    # Run Neighbor Joining
    newick = _neighbor_joining(d)
    
    return newick
