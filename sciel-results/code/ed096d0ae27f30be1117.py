"""Phylogenetic topology recovery via NJ with vectorized distance computation."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np


def _compute_distance_matrix(sequences, labels):
    """Compute pairwise JC-corrected distances using vectorized operations."""
    n = len(labels)
    if n <= 1:
        return np.zeros((n, n), dtype=np.float64)
    
    L = len(sequences[labels[0]])
    
    # Convert sequences to numpy arrays for vectorized comparison
    # Encode A=0, C=1, G=2, T=3
    mat = np.zeros((n, L), dtype=np.int8)
    for i, lab in enumerate(labels):
        s = sequences[lab]
        for j, c in enumerate(s):
            if c == 'A':
                mat[i, j] = 0
            elif c == 'C':
                mat[i, j] = 1
            elif c == 'G':
                mat[i, j] = 2
            else:  # 'T'
                mat[i, j] = 3

    # Compute pairwise differences using broadcasting
    # For memory efficiency, process in chunks if needed
    dist = np.zeros((n, n), dtype=np.float64)
    
    # For n <= 32, we can do the full broadcast at once
    # mat[:, None, :] has shape (n, 1, L)
    # mat[None, :, :] has shape (1, n, L)
    # Result: (n, n, L) boolean array
    # Memory: 32*32*2000*1 byte = 2MB, which is fine
    
    diff = (mat[:, np.newaxis, :] != mat[np.newaxis, :, :])  # (n, n, L)
    counts = diff.sum(axis=2).astype(np.float64)  # (n, n)
    p = counts / L
    
    # Apply JC correction
    # Handle p >= 0.7499 to avoid log(0) or negative argument
    p_safe = np.clip(p, 0, 0.7499)
    jc = -0.75 * np.log(1.0 - 4.0 * p_safe / 3.0)
    jc[p >= 0.7499] = 3.0
    
    np.fill_diagonal(jc, 0.0)
    return jc


def _nj_newick(dist, labels):
    """
    Neighbor Joining algorithm returning a Newick string.
    dist: n x n distance matrix
    labels: list of n labels
    """
    n = len(labels)
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]})" + ";"

    # Work on a copy of the distance matrix
    D = dist.copy()
    
    # nodes: list of current node indices (initially 0..n-1, then new nodes get IDs n, n+1, ...)
    nodes = list(range(n))
    # trees: dict mapping node index to Newick substring
    trees = {i: labels[i] for i in range(n)}
    next_id = n

    while len(nodes) > 3:
        m = len(nodes)
        
        # Compute row sums (Q matrix row sums)
        row_sum = np.zeros(m, dtype=np.float64)
        for idx, i in enumerate(nodes):
            for jdx, j in enumerate(nodes):
                if i != j:
                    row_sum[idx] += D[i, j]

        # Find the pair (a, b) that minimizes the NJ criterion
        # Q(i,j) = (m-2)*d(i,j) - r(i) - r(j)
        best_criterion = float('inf')
        best_a = None
        best_b = None
        best_a_idx = -1
        best_b_idx = -1
        
        for idx_i in range(m):
            for idx_j in range(idx_i + 1, m):
                i = nodes[idx_i]
                j = nodes[idx_j]
                criterion = (m - 2) * D[i, j] - row_sum[idx_i] - row_sum[idx_j]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_a = i
                    best_b = j
                    best_a_idx = idx_i
                    best_b_idx = idx_j

        if best_a is None or best_b is None:
            # Fallback: star tree
            return "(" + ",".join(trees[x] for x in nodes) + ");"

        a = best_a
        b = best_b
        d_ab = D[a, b]
        
        # Create new node
        w = next_id
        next_id += 1

        # Update trees
        trees[w] = f"({trees[a]},{trees[b]})"
        
        # Update distance matrix for the new node
        # d(w, x) = 0.5 * (d(a, x) + d(b, x) - d(a, b))
        for x in nodes:
            if x == a or x == b:
                continue
            D[w, x] = 0.5 * (D[a, x] + D[b, x] - d_ab)
            D[x, w] = D[w, x]
        
        # Remove a and b, add w
        nodes = [x for x in nodes if x != a and x != b]
        nodes.append(w)

    # Final step: 3 nodes remain
    # In an unrooted tree with 3 taxa, it's a star
    a, b, c = nodes
    return f"({trees[a]},{trees[b]},{trees[c]})" + ";"


def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 170  # Safety margin for 180s budget
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n == 0:
        return ""
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]})" + ";"

    # 1. Compute distance matrix
    dist_matrix = _compute_distance_matrix(sequences, labels)
    
    # Check time
    if time.time() > deadline:
        # Fallback: return a star tree
        return "(" + ",".join(labels) + ");"
    
    # 2. Run NJ
    try:
        result = _nj_newick(dist_matrix, labels)
    except Exception:
        # Fallback: return a star tree
        result = "(" + ",".join(labels) + ");"
    
    return result
# EVOLVE-BLOCK-END
